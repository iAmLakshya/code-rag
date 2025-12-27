from dataclasses import dataclass, field

from code_rag.query.graph_reasoning import GraphNode
from code_rag.query.query_planner import QueryIntent

MAX_CODE_SNIPPET_LENGTH = 3000
MAX_CONTEXT_ENTITIES = 20
MAX_RELATED_CODE_SNIPPETS = 5


@dataclass
class CodeSnippet:
    content: str
    file_path: str
    start_line: int
    end_line: int
    entity_name: str
    entity_type: str
    language: str | None = None
    relevance_score: float = 0.0


@dataclass
class EntityContext:
    entity: GraphNode
    code_snippet: CodeSnippet | None = None
    implementation_summary: str | None = None
    caller_summaries: list[str] = field(default_factory=list)
    callee_summaries: list[str] = field(default_factory=list)
    related_entities: list[str] = field(default_factory=list)


@dataclass
class EnrichedContext:
    query: str
    intent: QueryIntent
    primary_contexts: list[EntityContext]
    call_chain_explanations: list[str]
    hierarchy_explanations: list[str]
    file_summaries: dict[str, str]
    dependency_map: dict[str, list[str]]
    code_snippets: list[CodeSnippet]
    graph_summary: str
    total_entities_found: int
    reasoning_notes: list[str]
