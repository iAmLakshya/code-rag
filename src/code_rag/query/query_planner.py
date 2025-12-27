import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from code_rag.config import get_settings
from code_rag.core.errors import QueryError

logger = logging.getLogger(__name__)

_RE_CODE_BLOCK_START = re.compile(r'^```(?:json)?\s*\n?', re.MULTILINE)
_RE_CODE_BLOCK_END = re.compile(r'\n?```\s*$', re.MULTILINE)
_RE_JSON_OBJECT = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)
_RE_CAMEL_CASE = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b")
_RE_SNAKE_CASE = re.compile(r"\b([a-z]+(?:_[a-z]+)+)\b")
_RE_BACKTICK = re.compile(r"`([^`]+)`")


class QueryIntent(Enum):
    FIND_CALLERS = "find_callers"
    FIND_CALLEES = "find_callees"  # What does X call?
    FIND_CALL_CHAIN = "find_call_chain"  # How does A eventually call B?
    FIND_HIERARCHY = "find_hierarchy"  # What extends/implements X?
    FIND_IMPLEMENTATIONS = "find_implementations"  # Where is X implemented?
    FIND_USAGES = "find_usages"  # Where is X used?
    FIND_DEPENDENCIES = "find_dependencies"  # What does X depend on?
    FIND_DEPENDENTS = "find_dependents"  # What depends on X?

    # Navigational intents
    LOCATE_ENTITY = "locate_entity"  # Where is X defined?
    LOCATE_FILE = "locate_file"  # Find file containing X

    # Explanatory intents - require rich context
    EXPLAIN_IMPLEMENTATION = "explain_implementation"  # How does X work?
    EXPLAIN_RELATIONSHIP = "explain_relationship"  # How does A relate to B?
    EXPLAIN_DATA_FLOW = "explain_data_flow"  # How does data flow through X?
    EXPLAIN_ARCHITECTURE = "explain_architecture"  # How is X architected?

    # Semantic intents - vector-primary
    FIND_SIMILAR = "find_similar"  # Find code similar to X
    SEARCH_FUNCTIONALITY = "search_functionality"  # Find code that does X
    SEARCH_PATTERN = "search_pattern"  # Find pattern/approach for X


@dataclass
class ExtractedEntity:
    """An entity extracted from the query."""

    name: str
    entity_type: str | None = None  # class, function, method, file, module
    is_primary: bool = False  # Is this the main subject of the query?
    context: str | None = None  # Additional context about the entity


@dataclass
class QueryRelationship:
    """A relationship between entities in the query."""

    source: str
    target: str
    relationship_type: str  # calls, extends, imports, uses, etc.


@dataclass
class SubQuery:
    """A decomposed sub-query for execution."""

    query_text: str
    intent: QueryIntent
    entities: list[ExtractedEntity]
    relationships: list[QueryRelationship]
    search_type: str  # "graph", "vector", "hybrid"
    priority: int = 1  # Lower is higher priority
    depends_on: list[int] = field(default_factory=list)  # Indices of dependent sub-queries


@dataclass
class QueryPlan:
    """A complete query execution plan."""

    original_query: str
    primary_intent: QueryIntent
    sub_queries: list[SubQuery]
    entities: list[ExtractedEntity]
    relationships: list[QueryRelationship]
    requires_multi_hop: bool = False
    max_hops: int = 1
    context_requirements: list[str] = field(default_factory=list)
    reasoning: str = ""


QUERY_ANALYSIS_PROMPT = """You are an expert at analyzing code-related questions to understand what information is needed to answer them.

Analyze the following question about a codebase and extract:
1. The PRIMARY INTENT - what is the user fundamentally trying to understand?
2. ENTITIES mentioned (class names, function names, file names, modules, concepts)
3. RELATIONSHIPS between entities (calls, extends, imports, uses, contains)
4. Whether MULTI-HOP reasoning is needed (e.g., "what calls functions that call X" requires 2 hops)
5. What CONTEXT is needed to fully answer (implementation details, call chains, data flow, etc.)

Respond with a JSON object with this structure:
{{
    "primary_intent": "one of: find_callers, find_callees, find_call_chain, find_hierarchy, find_implementations, find_usages, find_dependencies, find_dependents, locate_entity, locate_file, explain_implementation, explain_relationship, explain_data_flow, explain_architecture, find_similar, search_functionality, search_pattern",
    "entities": [
        {{"name": "EntityName", "type": "class|function|method|file|module|concept", "is_primary": true, "context": "optional context"}}
    ],
    "relationships": [
        {{"source": "Entity1", "target": "Entity2", "type": "calls|extends|imports|uses|contains|related_to"}}
    ],
    "multi_hop": {{
        "required": true|false,
        "max_hops": 1-5,
        "reasoning": "why multi-hop is needed"
    }},
    "context_requirements": ["implementation_details", "call_chain", "data_flow", "file_context", "dependencies", "usages"],
    "sub_queries": [
        {{
            "query": "sub-query text",
            "intent": "intent type",
            "search_type": "graph|vector|hybrid",
            "priority": 1,
            "depends_on": []
        }}
    ],
    "reasoning": "explanation of how to answer this query"
}}

Question: {question}

Respond ONLY with valid JSON, no additional text."""


class QueryPlanner:
    """LLM-powered query planner for semantic understanding and decomposition."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        settings = get_settings()
        self.model = model or settings.llm_model
        self._client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def plan_query(self, question: str) -> QueryPlan:
        if not question or not question.strip():
            raise QueryError("Question cannot be empty")

        try:
            logger.debug(f"Planning query: {question}")

            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing code questions. Always respond with valid JSON only. No markdown, no explanations, just the JSON object.",
                    },
                    {
                        "role": "user",
                        "content": QUERY_ANALYSIS_PROMPT.format(question=question),
                    },
                ],
                temperature=0.0,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if content:
                content = content.strip()

            analysis = self._extract_json(content)
            plan = self._build_query_plan(question, analysis)

            logger.debug(f"Query plan created: intent={plan.primary_intent}, sub_queries={len(plan.sub_queries)}")
            return plan

        except (json.JSONDecodeError, ValueError, KeyError, TypeError, AttributeError) as e:
            logger.debug(f"Query planning parse error ({type(e).__name__}), using heuristic fallback")
            return self._fallback_plan(question)
        except Exception as e:
            logger.debug(
                f"Unexpected error during query planning ({type(e).__name__}: {e}), "
                "using heuristic fallback"
            )
            return self._fallback_plan(question)

    def _extract_json(self, content: str) -> dict[str, Any]:
        if not content:
            raise json.JSONDecodeError("Empty content", "", 0)

        content = content.strip()

        def validate_dict(result: Any) -> dict[str, Any]:
            if not isinstance(result, dict):
                raise ValueError(
                    f"Expected JSON object but got {type(result).__name__}: "
                    f"{str(result)[:50]}..."
                )
            return result

        try:
            return validate_dict(json.loads(content))
        except json.JSONDecodeError:
            pass

        content = _RE_CODE_BLOCK_START.sub('', content)
        content = _RE_CODE_BLOCK_END.sub('', content)
        content = content.strip()

        try:
            return validate_dict(json.loads(content))
        except json.JSONDecodeError:
            pass

        first_brace = content.find('{')
        last_brace = content.rfind('}')

        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = content[first_brace:last_brace + 1]
            try:
                return validate_dict(json.loads(json_str))
            except json.JSONDecodeError:
                pass

        match = _RE_JSON_OBJECT.search(content)
        if match:
            try:
                return validate_dict(json.loads(match.group()))
            except json.JSONDecodeError:
                pass

        preview = content[:100] + "..." if len(content) > 100 else content
        raise json.JSONDecodeError(f"Could not extract JSON from: {preview}", content, 0)

    def _build_query_plan(self, question: str, analysis: dict[str, Any]) -> QueryPlan:
        intent_str = analysis.get("primary_intent", "search_functionality")
        try:
            primary_intent = QueryIntent(intent_str)
        except ValueError:
            primary_intent = QueryIntent.SEARCH_FUNCTIONALITY

        entities = []
        for e in analysis.get("entities", []):
            entities.append(
                ExtractedEntity(
                    name=e.get("name", ""),
                    entity_type=e.get("type"),
                    is_primary=e.get("is_primary", False),
                    context=e.get("context"),
                )
            )

        relationships = []
        for r in analysis.get("relationships", []):
            relationships.append(
                QueryRelationship(
                    source=r.get("source", ""),
                    target=r.get("target", ""),
                    relationship_type=r.get("type", "related_to"),
                )
            )

        multi_hop = analysis.get("multi_hop", {})
        requires_multi_hop = multi_hop.get("required", False)
        max_hops = multi_hop.get("max_hops", 1)

        context_requirements = analysis.get("context_requirements", [])

        sub_queries = []
        for i, sq in enumerate(analysis.get("sub_queries", [])):
            intent_str = sq.get("intent", "search_functionality")
            try:
                sq_intent = QueryIntent(intent_str)
            except ValueError:
                sq_intent = QueryIntent.SEARCH_FUNCTIONALITY

            sq_entities = [e for e in entities if e.name.lower() in sq.get("query", "").lower()]

            sub_queries.append(
                SubQuery(
                    query_text=sq.get("query", question),
                    intent=sq_intent,
                    entities=sq_entities,
                    relationships=[],
                    search_type=sq.get("search_type", "hybrid"),
                    priority=sq.get("priority", i + 1),
                    depends_on=sq.get("depends_on", []),
                )
            )

        if not sub_queries:
            sub_queries.append(
                SubQuery(
                    query_text=question,
                    intent=primary_intent,
                    entities=entities,
                    relationships=relationships,
                    search_type=self._determine_search_type(primary_intent),
                    priority=1,
                )
            )

        return QueryPlan(
            original_query=question,
            primary_intent=primary_intent,
            sub_queries=sub_queries,
            entities=entities,
            relationships=relationships,
            requires_multi_hop=requires_multi_hop,
            max_hops=max_hops,
            context_requirements=context_requirements,
            reasoning=analysis.get("reasoning", ""),
        )

    def _determine_search_type(self, intent: QueryIntent) -> str:
        graph_primary = {
            QueryIntent.FIND_CALLERS,
            QueryIntent.FIND_CALLEES,
            QueryIntent.FIND_CALL_CHAIN,
            QueryIntent.FIND_HIERARCHY,
            QueryIntent.FIND_USAGES,
            QueryIntent.FIND_DEPENDENCIES,
            QueryIntent.FIND_DEPENDENTS,
            QueryIntent.LOCATE_ENTITY,
            QueryIntent.LOCATE_FILE,
        }

        vector_primary = {
            QueryIntent.FIND_SIMILAR,
            QueryIntent.SEARCH_FUNCTIONALITY,
            QueryIntent.SEARCH_PATTERN,
        }

        if intent in graph_primary:
            return "graph"
        elif intent in vector_primary:
            return "vector"
        else:
            return "hybrid"

    def _fallback_plan(self, question: str) -> QueryPlan:
        logger.debug("Using fallback heuristic query planning")

        question_lower = question.lower()

        if any(kw in question_lower for kw in ["what calls", "who calls", "callers of"]):
            intent = QueryIntent.FIND_CALLERS
            search_type = "graph"
        elif any(kw in question_lower for kw in ["calls what", "what does.*call"]):
            intent = QueryIntent.FIND_CALLEES
            search_type = "graph"
        elif any(kw in question_lower for kw in ["call chain", "eventually call", "path from.*to"]):
            intent = QueryIntent.FIND_CALL_CHAIN
            search_type = "graph"
        elif any(kw in question_lower for kw in ["extends", "inherits", "subclass", "hierarchy"]):
            intent = QueryIntent.FIND_HIERARCHY
            search_type = "graph"
        elif any(kw in question_lower for kw in ["how does", "how is.*implemented", "implementation of"]):
            intent = QueryIntent.EXPLAIN_IMPLEMENTATION
            search_type = "hybrid"
        elif any(kw in question_lower for kw in ["where is", "find the", "locate"]):
            intent = QueryIntent.LOCATE_ENTITY
            search_type = "graph"
        elif any(kw in question_lower for kw in ["similar to", "like"]):
            intent = QueryIntent.FIND_SIMILAR
            search_type = "vector"
        else:
            intent = QueryIntent.SEARCH_FUNCTIONALITY
            search_type = "hybrid"

        entities = []
        for match in _RE_CAMEL_CASE.findall(question):
            entities.append(ExtractedEntity(name=match, entity_type="class", is_primary=True))
        for match in _RE_SNAKE_CASE.findall(question):
            entities.append(ExtractedEntity(name=match, entity_type="function", is_primary=True))
        for match in _RE_BACKTICK.findall(question):
            entities.append(ExtractedEntity(name=match, is_primary=True))

        requires_multi_hop = any(
            kw in question_lower
            for kw in ["eventually", "indirectly", "chain", "path", "through", "via"]
        )

        return QueryPlan(
            original_query=question,
            primary_intent=intent,
            sub_queries=[
                SubQuery(
                    query_text=question,
                    intent=intent,
                    entities=entities,
                    relationships=[],
                    search_type=search_type,
                    priority=1,
                )
            ],
            entities=entities,
            relationships=[],
            requires_multi_hop=requires_multi_hop,
            max_hops=3 if requires_multi_hop else 1,
            context_requirements=["implementation_details"] if intent == QueryIntent.EXPLAIN_IMPLEMENTATION else [],
            reasoning="Fallback heuristic analysis",
        )
