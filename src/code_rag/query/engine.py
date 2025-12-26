"""Hybrid query engine with advanced graph reasoning and context building.

This query engine combines:
- LLM-powered query planning and decomposition
- Multi-hop graph reasoning for structural queries
- Rich context building for comprehensive answers
- Intelligent hybrid ranking with graph centrality
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from code_rag.config import get_settings
from code_rag.core.errors import QueryError
from code_rag.embeddings.client import CollectionName, QdrantManager
from code_rag.embeddings.embedder import OpenAIEmbedder
from code_rag.graph.client import MemgraphClient
from code_rag.query.context_builder import ContextBuilder, EnrichedContext, format_context_for_llm
from code_rag.query.graph_reasoning import GraphContext, GraphReasoningEngine
from code_rag.query.hybrid_ranker import HybridRanker, RankedResult, RankingConfig
from code_rag.query.query_planner import QueryIntent, QueryPlan, QueryPlanner
from code_rag.query.responder import ResponseGenerator
from code_rag.query.vector_search import VectorSearcher

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_LIMIT = 15
MAX_VECTOR_RESULTS = 20
MAX_CENTRALITY_LOOKUPS = 10


@dataclass
class QueryResult:
    """Result from the query engine."""

    answer: str
    sources: list[RankedResult]
    query_plan: QueryPlan
    context: EnrichedContext
    graph_context: GraphContext
    execution_stats: dict[str, Any]


class QueryEngine:
    """Query engine with multi-hop graph reasoning and rich context.

    Features:
    1. LLM-powered query planning and decomposition
    2. Multi-hop graph traversals for structural queries
    3. Rich context building with implementation details
    4. Intelligent ranking with graph centrality
    5. Comprehensive answers with full code context
    """

    def __init__(
        self,
        memgraph: MemgraphClient | None = None,
        qdrant: QdrantManager | None = None,
        embedder: OpenAIEmbedder | None = None,
        planner: QueryPlanner | None = None,
        graph_engine: GraphReasoningEngine | None = None,
        vector_searcher: VectorSearcher | None = None,
        context_builder: ContextBuilder | None = None,
        ranker: HybridRanker | None = None,
        responder: ResponseGenerator | None = None,
    ):
        """Initialize the enhanced query engine.

        Args:
            memgraph: Memgraph client.
            qdrant: Qdrant manager.
            embedder: OpenAI embedder.
            planner: Query planner.
            graph_engine: Graph reasoning engine.
            vector_searcher: Vector searcher.
            context_builder: Context builder.
            ranker: Hybrid ranker.
            responder: Response generator.
        """
        self._memgraph = memgraph
        self._qdrant = qdrant
        self._embedder = embedder
        self._planner = planner
        self._graph_engine = graph_engine
        self._vector_searcher = vector_searcher
        self._context_builder = context_builder
        self._ranker = ranker or HybridRanker(RankingConfig())
        self._responder = responder
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        logger.info("Initializing query engine")

        # Initialize database clients
        if self._memgraph is None:
            self._memgraph = MemgraphClient()
            await self._memgraph.connect()

        if self._qdrant is None:
            self._qdrant = QdrantManager()
            await self._qdrant.connect()

        if self._embedder is None:
            self._embedder = OpenAIEmbedder()

        # Initialize query components
        if self._planner is None:
            self._planner = QueryPlanner()

        if self._graph_engine is None:
            self._graph_engine = GraphReasoningEngine(self._memgraph)

        if self._vector_searcher is None:
            self._vector_searcher = VectorSearcher(self._qdrant, self._embedder)

        if self._context_builder is None:
            self._context_builder = ContextBuilder(self._memgraph, self._qdrant)

        if self._responder is None:
            self._responder = ResponseGenerator()

        self._initialized = True
        logger.info("Query engine initialized")

    async def close(self) -> None:
        """Close all connections."""
        logger.info("Closing query engine")

        if self._memgraph:
            await self._memgraph.close()
        if self._qdrant:
            await self._qdrant.close()

        self._initialized = False

    async def query(
        self,
        question: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        language: str | None = None,
        use_llm_planning: bool = True,
    ) -> QueryResult:
        """Execute an enhanced query with multi-hop reasoning.

        Args:
            question: Natural language question.
            limit: Maximum results to return.
            language: Optional language filter.
            use_llm_planning: Whether to use LLM for query planning.

        Returns:
            QueryResult with answer, sources, and context.

        Raises:
            QueryError: If query execution fails.
        """
        await self.initialize()

        stats = {
            "planning_time_ms": 0,
            "graph_time_ms": 0,
            "vector_time_ms": 0,
            "ranking_time_ms": 0,
            "context_time_ms": 0,
            "response_time_ms": 0,
        }

        try:
            logger.info(f"Executing query: {question}")

            import time
            start = time.time()

            if use_llm_planning:
                plan = await self._planner.plan_query(question)
            else:
                plan = self._planner._fallback_plan(question)

            stats["planning_time_ms"] = int((time.time() - start) * 1000)
            logger.debug(f"Query plan: intent={plan.primary_intent}, entities={len(plan.entities)}")

            start = time.time()

            graph_task = self._graph_engine.execute_query_plan(plan)
            vector_task = self._execute_vector_search(question, plan, limit, language)

            graph_context, vector_results = await asyncio.gather(
                graph_task,
                vector_task,
                return_exceptions=True,
            )

            stats["graph_time_ms"] = int((time.time() - start) * 1000)

            if isinstance(graph_context, Exception):
                logger.warning(f"Graph search failed: {graph_context}")
                graph_context = GraphContext(
                    primary_entities=[],
                    callers=[],
                    callees=[],
                    parent_classes=[],
                    child_classes=[],
                    methods=[],
                    containing_class=None,
                    file_context=[],
                    dependencies=[],
                    dependents=[],
                    call_chains=[],
                    inheritance_chains=[],
                )

            if isinstance(vector_results, Exception):
                logger.warning(f"Vector search failed: {vector_results}")
                vector_results = []

            start = time.time()
            centrality_scores = await self._get_centrality_scores(graph_context, vector_results)
            stats["vector_time_ms"] = int((time.time() - start) * 1000)

            start = time.time()
            ranked_results = self._ranker.rank_results(
                plan,
                graph_context,
                vector_results,
                centrality_scores,
            )
            stats["ranking_time_ms"] = int((time.time() - start) * 1000)

            start = time.time()
            enriched_context = await self._context_builder.build_enriched_context(
                plan,
                graph_context,
                vector_results,
            )
            stats["context_time_ms"] = int((time.time() - start) * 1000)

            start = time.time()
            answer = await self._generate_enhanced_response(
                question,
                plan,
                ranked_results[:limit],
                enriched_context,
            )
            stats["response_time_ms"] = int((time.time() - start) * 1000)

            total_time = sum(stats.values())
            logger.info(
                f"Query completed in {total_time}ms: "
                f"{len(ranked_results)} results, intent={plan.primary_intent}"
            )

            return QueryResult(
                answer=answer,
                sources=ranked_results[:limit],
                query_plan=plan,
                context=enriched_context,
                graph_context=graph_context,
                execution_stats=stats,
            )

        except QueryError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during enhanced query: {e}")
            raise QueryError("Query execution failed", cause=e)

    async def search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        language: str | None = None,
        entity_type: str | None = None,
    ) -> list[RankedResult]:
        """Execute a search without generating a response.

        Args:
            query: Search query.
            limit: Maximum results.
            language: Optional language filter.
            entity_type: Optional entity type filter.

        Returns:
            List of ranked results.

        Raises:
            QueryError: If search fails.
        """
        await self.initialize()

        try:
            logger.info(f"Executing search: {query}")

            # Plan query
            plan = await self._planner.plan_query(query)

            # Execute searches in parallel
            graph_context, vector_results = await asyncio.gather(
                self._graph_engine.execute_query_plan(plan),
                self._execute_vector_search(query, plan, limit * 2, language),
            )

            # Get centrality scores
            centrality_scores = await self._get_centrality_scores(graph_context, vector_results)

            # Rank results
            ranked_results = self._ranker.rank_results(
                plan,
                graph_context,
                vector_results,
                centrality_scores,
            )

            logger.info(f"Search completed: {len(ranked_results[:limit])} results")
            return ranked_results[:limit]

        except QueryError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            raise QueryError("Search execution failed", cause=e)

    async def explain_entity(
        self,
        entity_name: str,
        include_callers: bool = True,
        include_callees: bool = True,
        max_depth: int = 2,
    ) -> QueryResult:
        """Get a comprehensive explanation of an entity.

        Args:
            entity_name: Name of the entity to explain.
            include_callers: Whether to include caller information.
            include_callees: Whether to include callee information.
            max_depth: Maximum depth for call graph traversal.

        Returns:
            QueryResult with comprehensive entity explanation.
        """
        question = f"Explain how {entity_name} works and is used in the codebase"
        return await self.query(question)

    async def find_call_path(
        self,
        source_name: str,
        target_name: str,
        max_hops: int = 5,
    ) -> QueryResult:
        """Find call paths between two entities.

        Args:
            source_name: Name of the source entity.
            target_name: Name of the target entity.
            max_hops: Maximum path length.

        Returns:
            QueryResult with call path information.
        """
        question = f"How does {source_name} eventually call {target_name}? Show the call chain."
        return await self.query(question)

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the indexed codebase.

        Returns:
            Statistics dictionary.
        """
        await self.initialize()

        try:
            # Get graph stats
            graph_stats = await self._memgraph.execute(
                """
                MATCH (f:File)
                WITH count(f) as file_count
                MATCH (c:Class)
                WITH file_count, count(c) as class_count
                MATCH (fn:Function)
                WITH file_count, class_count, count(fn) as function_count
                MATCH (m:Method)
                RETURN file_count, class_count, function_count, count(m) as method_count
                """
            )

            # Get vector stats
            try:
                code_info = await self._qdrant.get_collection_info(
                    CollectionName.CODE_CHUNKS.value
                )
                vector_count = code_info.points_count
            except Exception:
                vector_count = 0

            stats = graph_stats[0] if graph_stats else {}
            stats["vector_count"] = vector_count

            return stats

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            raise QueryError("Failed to get statistics", cause=e)

    async def _execute_vector_search(
        self,
        query: str,
        plan: QueryPlan,
        limit: int,
        language: str | None,
    ) -> list[dict[str, Any]]:
        """Execute vector search based on query plan.

        Args:
            query: Search query.
            plan: Query plan.
            limit: Maximum results.
            language: Language filter.

        Returns:
            Vector search results.
        """
        # Search code chunks
        code_results = await self._vector_searcher.search_code(
            query=query,
            limit=min(limit, MAX_VECTOR_RESULTS),
            language=language,
        )

        # For explanatory queries, also search summaries
        if plan.primary_intent in (
            QueryIntent.EXPLAIN_IMPLEMENTATION,
            QueryIntent.EXPLAIN_RELATIONSHIP,
            QueryIntent.EXPLAIN_DATA_FLOW,
            QueryIntent.EXPLAIN_ARCHITECTURE,
            QueryIntent.SEARCH_FUNCTIONALITY,
        ):
            summary_results = await self._vector_searcher.search_summaries(
                query=query,
                limit=limit // 2,
            )
            code_results.extend(summary_results)

        return code_results

    async def _get_centrality_scores(
        self,
        graph_context: GraphContext,
        vector_results: list[dict[str, Any]],
    ) -> dict[str, dict[str, int]]:
        """Get centrality scores for entities.

        Args:
            graph_context: Graph context.
            vector_results: Vector results.

        Returns:
            Dictionary mapping entity names to centrality scores.
        """
        entities = set()

        for entity in graph_context.primary_entities[:5]:
            entities.add(entity.qualified_name or entity.name)

        for vr in vector_results[:5]:
            if vr.get("entity_name"):
                entities.add(vr.get("graph_node_id") or vr.get("entity_name"))

        entities = list(entities)[:MAX_CENTRALITY_LOOKUPS]

        scores = {}
        if entities:
            tasks = [
                self._graph_engine.get_entity_centrality(name)
                for name in entities
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for name, result in zip(entities, results):
                if isinstance(result, dict):
                    scores[name] = result

        return scores

    async def _generate_enhanced_response(
        self,
        question: str,
        plan: QueryPlan,
        results: list[RankedResult],
        context: EnrichedContext,
    ) -> str:
        """Generate an enhanced response using full context.

        Args:
            question: Original question.
            plan: Query plan.
            results: Ranked results.
            context: Enriched context.

        Returns:
            Generated response.
        """
        context_text = format_context_for_llm(context)

        system_prompt = self._get_enhanced_system_prompt(plan.primary_intent)
        user_prompt = self._build_enhanced_user_prompt(question, plan, context_text)

        from openai import AsyncOpenAI
        settings = get_settings()
        client = AsyncOpenAI(api_key=settings.openai_api_key)

        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=settings.llm_temperature,
            max_tokens=2000,
        )

        return response.choices[0].message.content.strip()

    def _get_enhanced_system_prompt(self, intent: QueryIntent) -> str:
        """Get system prompt tailored to query intent.

        Args:
            intent: Query intent.

        Returns:
            System prompt.
        """
        base_prompt = """You are an expert code analyst helping developers understand codebases.
You have access to comprehensive context including:
- Code snippets and implementations
- Call graphs showing how functions interact
- Inheritance hierarchies
- File summaries and dependencies

When answering:
1. Be precise and reference specific files, functions, and line numbers
2. Explain relationships between code entities clearly
3. Use code snippets to illustrate your points
4. If information is incomplete, acknowledge what's known and what's uncertain
5. Structure complex explanations with clear headers and bullet points

Format your response using markdown for readability."""

        intent_additions = {
            QueryIntent.FIND_CALLERS: "\n\nFocus on explaining WHERE and HOW the entity is used. Show the call chain from callers.",
            QueryIntent.FIND_CALLEES: "\n\nFocus on explaining WHAT the entity depends on. Show what it calls and why.",
            QueryIntent.FIND_CALL_CHAIN: "\n\nFocus on the complete path between entities. Explain each step in the chain.",
            QueryIntent.FIND_HIERARCHY: "\n\nFocus on the inheritance structure. Explain what each level adds or overrides.",
            QueryIntent.EXPLAIN_IMPLEMENTATION: "\n\nProvide a deep dive into HOW the code works. Include implementation details.",
            QueryIntent.EXPLAIN_DATA_FLOW: "\n\nTrace how data moves through the system. Show transformations and handlers.",
            QueryIntent.SEARCH_FUNCTIONALITY: "\n\nExplain what the code DOES and how to use it.",
        }

        return base_prompt + intent_additions.get(intent, "")

    def _build_enhanced_user_prompt(
        self,
        question: str,
        plan: QueryPlan,
        context_text: str,
    ) -> str:
        """Build enhanced user prompt with full context.

        Args:
            question: Original question.
            plan: Query plan.
            context_text: Formatted context.

        Returns:
            User prompt.
        """
        prompt_parts = [
            f"## Question\n{question}\n",
            f"\n## Query Understanding\n",
            f"- Intent: {plan.primary_intent.value}\n",
        ]

        if plan.entities:
            entities_str = ", ".join(e.name for e in plan.entities)
            prompt_parts.append(f"- Key entities: {entities_str}\n")

        if plan.requires_multi_hop:
            prompt_parts.append(f"- Multi-hop reasoning required (up to {plan.max_hops} hops)\n")

        if plan.reasoning:
            prompt_parts.append(f"- Analysis: {plan.reasoning}\n")

        prompt_parts.append(f"\n{context_text}\n")
        prompt_parts.append("\n## Instructions\n")
        prompt_parts.append("Based on the above context, provide a comprehensive answer to the question. ")
        prompt_parts.append("Reference specific code, files, and line numbers where relevant.")

        return "".join(prompt_parts)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
