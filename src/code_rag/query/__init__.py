"""Query engine module with multi-hop graph reasoning.

This module provides:
- LLM-powered query planning and decomposition
- Multi-hop graph traversals for structural queries
- Rich context building with implementation details
- Intelligent hybrid ranking with graph centrality
"""

from code_rag.query.engine import QueryEngine, QueryResult
from code_rag.query.graph_search import GraphSearcher
from code_rag.query.graph_reasoning import (
    GraphContext,
    GraphNode,
    GraphPath,
    GraphReasoningEngine,
)
from code_rag.query.query_planner import (
    ExtractedEntity,
    QueryIntent,
    QueryPlan,
    QueryPlanner,
    SubQuery,
)
from code_rag.query.context_builder import (
    CodeSnippet,
    ContextBuilder,
    EnrichedContext,
    EntityContext,
    format_context_for_llm,
)
from code_rag.query.hybrid_ranker import (
    HybridRanker,
    RankedResult,
    RankingConfig,
    RankingSignal,
)
from code_rag.query.reranker import ResultReranker, SearchResult
from code_rag.query.responder import ResponseGenerator
from code_rag.query.vector_search import VectorSearcher

__all__ = [
    # Main engine
    "QueryEngine",
    "QueryResult",
    # Query planning
    "QueryPlanner",
    "QueryPlan",
    "QueryIntent",
    "ExtractedEntity",
    "SubQuery",
    # Graph reasoning
    "GraphReasoningEngine",
    "GraphContext",
    "GraphNode",
    "GraphPath",
    "GraphSearcher",
    # Context building
    "ContextBuilder",
    "EnrichedContext",
    "EntityContext",
    "CodeSnippet",
    "format_context_for_llm",
    # Ranking
    "HybridRanker",
    "RankedResult",
    "RankingConfig",
    "RankingSignal",
    # Legacy components
    "VectorSearcher",
    "ResponseGenerator",
    "ResultReranker",
    "SearchResult",
]
