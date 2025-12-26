"""Intelligent hybrid ranking with graph centrality and query-aware scoring."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from code_rag.core.types import ResultSource
from code_rag.query.graph_reasoning import GraphContext, GraphNode
from code_rag.query.query_planner import QueryIntent, QueryPlan

logger = logging.getLogger(__name__)

# Default weights
DEFAULT_GRAPH_WEIGHT = 0.5
DEFAULT_VECTOR_WEIGHT = 0.5
DEFAULT_CENTRALITY_WEIGHT = 0.2
DEFAULT_CONTEXT_WEIGHT = 0.1

# Limits
MAX_RESULTS_PER_FILE = 5
MAX_TOTAL_RESULTS = 50


class RankingSignal(Enum):
    """Signals used in ranking computation."""

    GRAPH_MATCH = "graph_match"
    VECTOR_SIMILARITY = "vector_similarity"
    CENTRALITY = "centrality"
    QUERY_ENTITY_MATCH = "query_entity_match"
    RELATIONSHIP_RELEVANCE = "relationship_relevance"
    CODE_QUALITY = "code_quality"
    CONTEXT_RICHNESS = "context_richness"


@dataclass
class RankedResult:
    """A search result with detailed ranking information."""

    # Core identification
    file_path: str
    entity_name: str
    entity_type: str
    qualified_name: str | None = None

    # Content
    content: str | None = None
    summary: str | None = None
    signature: str | None = None
    docstring: str | None = None

    # Location
    start_line: int | None = None
    end_line: int | None = None

    # Source tracking
    source: str = ResultSource.HYBRID.value
    graph_node_id: str | None = None

    # Scoring
    final_score: float = 0.0
    signal_scores: dict[str, float] = field(default_factory=dict)

    # Context
    callers: list[str] = field(default_factory=list)
    callees: list[str] = field(default_factory=list)
    depth_from_query: int | None = None
    relationship_path: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_key(self) -> str:
        """Generate unique key for deduplication."""
        return f"{self.file_path}:{self.entity_name}:{self.start_line}"


@dataclass
class RankingConfig:
    """Configuration for the ranking algorithm."""

    # Source weights
    graph_weight: float = DEFAULT_GRAPH_WEIGHT
    vector_weight: float = DEFAULT_VECTOR_WEIGHT

    # Signal weights
    centrality_weight: float = DEFAULT_CENTRALITY_WEIGHT
    context_weight: float = DEFAULT_CONTEXT_WEIGHT
    entity_match_bonus: float = 0.3
    relationship_bonus: float = 0.15

    # Query-type specific adjustments
    query_type_adjustments: dict[QueryIntent, dict[str, float]] = field(default_factory=dict)

    # Limits
    max_per_file: int = MAX_RESULTS_PER_FILE
    max_total: int = MAX_TOTAL_RESULTS

    def __post_init__(self):
        """Initialize query-type adjustments."""
        if not self.query_type_adjustments:
            self.query_type_adjustments = {
                # Structural queries - boost graph results
                QueryIntent.FIND_CALLERS: {"graph_weight": 0.8, "vector_weight": 0.2},
                QueryIntent.FIND_CALLEES: {"graph_weight": 0.8, "vector_weight": 0.2},
                QueryIntent.FIND_CALL_CHAIN: {"graph_weight": 0.9, "vector_weight": 0.1},
                QueryIntent.FIND_HIERARCHY: {"graph_weight": 0.85, "vector_weight": 0.15},
                QueryIntent.FIND_USAGES: {"graph_weight": 0.7, "vector_weight": 0.3},
                QueryIntent.FIND_DEPENDENCIES: {"graph_weight": 0.75, "vector_weight": 0.25},

                # Navigational queries - balanced
                QueryIntent.LOCATE_ENTITY: {"graph_weight": 0.6, "vector_weight": 0.4},
                QueryIntent.LOCATE_FILE: {"graph_weight": 0.5, "vector_weight": 0.5},

                # Explanatory queries - need both
                QueryIntent.EXPLAIN_IMPLEMENTATION: {"graph_weight": 0.5, "vector_weight": 0.5},
                QueryIntent.EXPLAIN_RELATIONSHIP: {"graph_weight": 0.6, "vector_weight": 0.4},
                QueryIntent.EXPLAIN_DATA_FLOW: {"graph_weight": 0.65, "vector_weight": 0.35},

                # Semantic queries - boost vector results
                QueryIntent.FIND_SIMILAR: {"graph_weight": 0.2, "vector_weight": 0.8},
                QueryIntent.SEARCH_FUNCTIONALITY: {"graph_weight": 0.3, "vector_weight": 0.7},
                QueryIntent.SEARCH_PATTERN: {"graph_weight": 0.25, "vector_weight": 0.75},
            }


class HybridRanker:
    """Intelligent hybrid ranking engine with multi-signal scoring."""

    def __init__(self, config: RankingConfig | None = None):
        """Initialize the ranker.

        Args:
            config: Ranking configuration. Uses defaults if not provided.
        """
        self.config = config or RankingConfig()

    def rank_results(
        self,
        plan: QueryPlan,
        graph_context: GraphContext,
        vector_results: list[dict[str, Any]],
        centrality_scores: dict[str, dict[str, int]] | None = None,
    ) -> list[RankedResult]:
        """Rank and merge results from graph and vector search.

        Args:
            plan: Query plan with intent and entities.
            graph_context: Context from graph reasoning.
            vector_results: Results from vector search.
            centrality_scores: Optional centrality scores for entities.

        Returns:
            Ranked and deduplicated results.
        """
        logger.debug(f"Ranking results: graph={len(self._count_graph_entities(graph_context))}, vector={len(vector_results)}")

        # Get adjusted weights for this query type
        weights = self._get_adjusted_weights(plan.primary_intent)

        # Build results map for deduplication and merging
        results_map: dict[str, RankedResult] = {}

        # Process graph results
        self._process_graph_results(
            results_map,
            graph_context,
            plan,
            weights,
            centrality_scores or {},
        )

        # Process vector results
        self._process_vector_results(
            results_map,
            vector_results,
            plan,
            weights,
            centrality_scores or {},
        )

        # Convert to list and sort by final score
        results = list(results_map.values())
        results.sort(key=lambda r: r.final_score, reverse=True)

        # Apply deduplication and limits
        deduplicated = self._deduplicate_and_limit(results)

        logger.debug(f"Ranking complete: {len(deduplicated)} results after deduplication")

        return deduplicated

    def _get_adjusted_weights(self, intent: QueryIntent) -> dict[str, float]:
        """Get weights adjusted for query type.

        Args:
            intent: Query intent.

        Returns:
            Dictionary of adjusted weights.
        """
        base_weights = {
            "graph_weight": self.config.graph_weight,
            "vector_weight": self.config.vector_weight,
            "centrality_weight": self.config.centrality_weight,
            "context_weight": self.config.context_weight,
        }

        # Apply query-type specific adjustments
        if intent in self.config.query_type_adjustments:
            adjustments = self.config.query_type_adjustments[intent]
            base_weights.update(adjustments)

        return base_weights

    def _process_graph_results(
        self,
        results_map: dict[str, RankedResult],
        graph_context: GraphContext,
        plan: QueryPlan,
        weights: dict[str, float],
        centrality_scores: dict[str, dict[str, int]],
    ) -> None:
        """Process and score graph results.

        Args:
            results_map: Map to add results to.
            graph_context: Graph context with entities.
            plan: Query plan.
            weights: Ranking weights.
            centrality_scores: Centrality scores.
        """
        query_entities = {e.name.lower() for e in plan.entities}

        # Process primary entities (highest relevance)
        for node in graph_context.primary_entities:
            result = self._graph_node_to_result(node)
            self._score_graph_result(
                result,
                weights,
                centrality_scores,
                query_entities,
                is_primary=True,
            )
            self._merge_into_map(results_map, result)

        # Process callers
        for node in graph_context.callers:
            result = self._graph_node_to_result(node)
            result.relationship_path = "caller"
            result.depth_from_query = node.metadata.get("depth", 1) if node.metadata else 1
            self._score_graph_result(
                result,
                weights,
                centrality_scores,
                query_entities,
                is_caller=True,
            )
            self._merge_into_map(results_map, result)

        # Process callees
        for node in graph_context.callees:
            result = self._graph_node_to_result(node)
            result.relationship_path = "callee"
            result.depth_from_query = node.metadata.get("depth", 1) if node.metadata else 1
            self._score_graph_result(
                result,
                weights,
                centrality_scores,
                query_entities,
                is_callee=True,
            )
            self._merge_into_map(results_map, result)

        # Process methods (for class queries)
        for node in graph_context.methods:
            result = self._graph_node_to_result(node)
            result.relationship_path = "method"
            self._score_graph_result(
                result,
                weights,
                centrality_scores,
                query_entities,
            )
            self._merge_into_map(results_map, result)

        # Process hierarchy
        for node in graph_context.parent_classes:
            result = self._graph_node_to_result(node)
            result.relationship_path = "parent_class"
            self._score_graph_result(
                result,
                weights,
                centrality_scores,
                query_entities,
            )
            self._merge_into_map(results_map, result)

        for node in graph_context.child_classes:
            result = self._graph_node_to_result(node)
            result.relationship_path = "child_class"
            self._score_graph_result(
                result,
                weights,
                centrality_scores,
                query_entities,
            )
            self._merge_into_map(results_map, result)

    def _process_vector_results(
        self,
        results_map: dict[str, RankedResult],
        vector_results: list[dict[str, Any]],
        plan: QueryPlan,
        weights: dict[str, float],
        centrality_scores: dict[str, dict[str, int]],
    ) -> None:
        """Process and score vector results.

        Args:
            results_map: Map to add results to.
            vector_results: Vector search results.
            plan: Query plan.
            weights: Ranking weights.
            centrality_scores: Centrality scores.
        """
        query_entities = {e.name.lower() for e in plan.entities}

        for vr in vector_results:
            result = self._vector_result_to_ranked(vr)
            self._score_vector_result(
                result,
                vr.get("score", 0.0),
                weights,
                centrality_scores,
                query_entities,
            )
            self._merge_into_map(results_map, result)

    def _score_graph_result(
        self,
        result: RankedResult,
        weights: dict[str, float],
        centrality_scores: dict[str, dict[str, int]],
        query_entities: set[str],
        is_primary: bool = False,
        is_caller: bool = False,
        is_callee: bool = False,
    ) -> None:
        """Compute score for a graph result.

        Args:
            result: Result to score.
            weights: Ranking weights.
            centrality_scores: Centrality scores.
            query_entities: Entities from query.
            is_primary: Whether this is a primary entity.
            is_caller: Whether this is a caller.
            is_callee: Whether this is a callee.
        """
        signals = {}

        # Base graph match score
        base_score = 1.0
        if is_primary:
            base_score = 1.0
        elif is_caller or is_callee:
            # Reduce score based on depth
            depth = result.depth_from_query or 1
            base_score = max(0.3, 1.0 - (depth - 1) * 0.2)

        signals[RankingSignal.GRAPH_MATCH.value] = base_score

        # Query entity match bonus
        entity_match = 0.0
        if result.entity_name.lower() in query_entities:
            entity_match = 1.0
        elif any(qe in result.entity_name.lower() for qe in query_entities):
            entity_match = 0.5
        signals[RankingSignal.QUERY_ENTITY_MATCH.value] = entity_match

        # Relationship relevance
        rel_score = 0.0
        if is_primary:
            rel_score = 1.0
        elif is_caller:
            rel_score = 0.8
        elif is_callee:
            rel_score = 0.7
        else:
            rel_score = 0.5
        signals[RankingSignal.RELATIONSHIP_RELEVANCE.value] = rel_score

        # Centrality score
        centrality_score = 0.0
        entity_key = result.qualified_name or result.entity_name
        if entity_key in centrality_scores:
            scores = centrality_scores[entity_key]
            total_degree = scores.get("total_degree", 0)
            # Normalize: assume max degree is around 50
            centrality_score = min(1.0, total_degree / 50)
        signals[RankingSignal.CENTRALITY.value] = centrality_score

        # Context richness
        context_score = 0.0
        if result.summary:
            context_score += 0.3
        if result.docstring:
            context_score += 0.2
        if result.signature:
            context_score += 0.2
        if result.content:
            context_score += 0.3
        signals[RankingSignal.CONTEXT_RICHNESS.value] = context_score

        # Compute final score
        final_score = (
            signals[RankingSignal.GRAPH_MATCH.value] * weights["graph_weight"]
            + signals.get(RankingSignal.QUERY_ENTITY_MATCH.value, 0) * self.config.entity_match_bonus
            + signals.get(RankingSignal.RELATIONSHIP_RELEVANCE.value, 0) * self.config.relationship_bonus
            + signals.get(RankingSignal.CENTRALITY.value, 0) * weights["centrality_weight"]
            + signals.get(RankingSignal.CONTEXT_RICHNESS.value, 0) * weights["context_weight"]
        )

        result.final_score = final_score
        result.signal_scores = signals
        result.source = ResultSource.GRAPH.value

    def _score_vector_result(
        self,
        result: RankedResult,
        vector_score: float,
        weights: dict[str, float],
        centrality_scores: dict[str, dict[str, int]],
        query_entities: set[str],
    ) -> None:
        """Compute score for a vector result.

        Args:
            result: Result to score.
            vector_score: Original vector similarity score.
            weights: Ranking weights.
            centrality_scores: Centrality scores.
            query_entities: Entities from query.
        """
        signals = {}

        # Vector similarity score (already normalized 0-1)
        signals[RankingSignal.VECTOR_SIMILARITY.value] = vector_score

        # Query entity match bonus
        entity_match = 0.0
        if result.entity_name.lower() in query_entities:
            entity_match = 1.0
        elif any(qe in result.entity_name.lower() for qe in query_entities):
            entity_match = 0.5
        signals[RankingSignal.QUERY_ENTITY_MATCH.value] = entity_match

        # Centrality score
        centrality_score = 0.0
        entity_key = result.qualified_name or result.entity_name
        if entity_key in centrality_scores:
            scores = centrality_scores[entity_key]
            total_degree = scores.get("total_degree", 0)
            centrality_score = min(1.0, total_degree / 50)
        signals[RankingSignal.CENTRALITY.value] = centrality_score

        # Code quality signals
        quality_score = 0.0
        if result.content:
            content_len = len(result.content)
            # Prefer medium-length snippets (not too short, not too long)
            if 100 < content_len < 2000:
                quality_score = 0.8
            elif 50 < content_len < 3000:
                quality_score = 0.5
            else:
                quality_score = 0.3
        signals[RankingSignal.CODE_QUALITY.value] = quality_score

        # Compute final score
        final_score = (
            signals[RankingSignal.VECTOR_SIMILARITY.value] * weights["vector_weight"]
            + signals.get(RankingSignal.QUERY_ENTITY_MATCH.value, 0) * self.config.entity_match_bonus
            + signals.get(RankingSignal.CENTRALITY.value, 0) * weights["centrality_weight"]
            + signals.get(RankingSignal.CODE_QUALITY.value, 0) * 0.1
        )

        result.final_score = final_score
        result.signal_scores = signals
        result.source = ResultSource.VECTOR.value

    def _merge_into_map(
        self,
        results_map: dict[str, RankedResult],
        result: RankedResult,
    ) -> None:
        """Merge a result into the results map, combining duplicates.

        Args:
            results_map: Map to merge into.
            result: Result to merge.
        """
        key = result.get_key()

        if key not in results_map:
            results_map[key] = result
        else:
            existing = results_map[key]

            # Combine scores (hybrid)
            combined_score = (existing.final_score + result.final_score) / 2
            combined_score *= 1.1  # Boost for appearing in both sources

            # Take the better content
            if not existing.content and result.content:
                existing.content = result.content
            if not existing.summary and result.summary:
                existing.summary = result.summary
            if not existing.signature and result.signature:
                existing.signature = result.signature
            if not existing.docstring and result.docstring:
                existing.docstring = result.docstring

            # Merge signal scores
            for signal, score in result.signal_scores.items():
                if signal in existing.signal_scores:
                    existing.signal_scores[signal] = max(existing.signal_scores[signal], score)
                else:
                    existing.signal_scores[signal] = score

            existing.final_score = combined_score
            existing.source = ResultSource.HYBRID.value

    def _deduplicate_and_limit(self, results: list[RankedResult]) -> list[RankedResult]:
        """Deduplicate results and apply limits.

        Args:
            results: Sorted list of results.

        Returns:
            Deduplicated and limited results.
        """
        seen_keys = set()
        file_counts: dict[str, int] = {}
        deduplicated = []

        for result in results:
            key = result.get_key()

            # Skip exact duplicates
            if key in seen_keys:
                continue

            # Apply per-file limit
            file_count = file_counts.get(result.file_path, 0)
            if file_count >= self.config.max_per_file:
                continue

            seen_keys.add(key)
            file_counts[result.file_path] = file_count + 1
            deduplicated.append(result)

            # Apply total limit
            if len(deduplicated) >= self.config.max_total:
                break

        return deduplicated

    def _graph_node_to_result(self, node: GraphNode) -> RankedResult:
        """Convert a GraphNode to a RankedResult.

        Args:
            node: Graph node.

        Returns:
            RankedResult.
        """
        return RankedResult(
            file_path=node.file_path,
            entity_name=node.name,
            entity_type=node.node_type,
            qualified_name=node.qualified_name,
            summary=node.summary,
            signature=node.signature,
            docstring=node.docstring,
            start_line=node.start_line,
            end_line=node.end_line,
            graph_node_id=node.qualified_name,
            metadata=node.metadata,
        )

    def _vector_result_to_ranked(self, vr: dict[str, Any]) -> RankedResult:
        """Convert a vector result to a RankedResult.

        Args:
            vr: Vector result dictionary.

        Returns:
            RankedResult.
        """
        return RankedResult(
            file_path=vr.get("file_path", ""),
            entity_name=vr.get("entity_name", ""),
            entity_type=vr.get("entity_type", ""),
            qualified_name=vr.get("graph_node_id"),
            content=vr.get("content"),
            summary=vr.get("summary"),
            start_line=vr.get("start_line"),
            end_line=vr.get("end_line"),
            graph_node_id=vr.get("graph_node_id"),
        )

    def _count_graph_entities(self, graph_context: GraphContext) -> list[GraphNode]:
        """Count total entities in graph context.

        Args:
            graph_context: Graph context.

        Returns:
            List of all entities.
        """
        entities = []
        entities.extend(graph_context.primary_entities)
        entities.extend(graph_context.callers)
        entities.extend(graph_context.callees)
        entities.extend(graph_context.methods)
        entities.extend(graph_context.parent_classes)
        entities.extend(graph_context.child_classes)
        return entities


def ranked_results_to_search_results(results: list[RankedResult]) -> list[dict[str, Any]]:
    """Convert RankedResults to the format expected by ResponseGenerator.

    Args:
        results: List of RankedResult objects.

    Returns:
        List of dictionaries compatible with SearchResult.
    """
    return [
        {
            "source": result.source,
            "score": result.final_score,
            "file_path": result.file_path,
            "entity_type": result.entity_type,
            "entity_name": result.entity_name,
            "content": result.content,
            "summary": result.summary,
            "start_line": result.start_line,
            "end_line": result.end_line,
            "graph_node_id": result.graph_node_id,
            "metadata": {
                "signal_scores": result.signal_scores,
                "relationship_path": result.relationship_path,
                "depth_from_query": result.depth_from_query,
                "signature": result.signature,
                "docstring": result.docstring,
                "callers": result.callers,
                "callees": result.callees,
            },
        }
        for result in results
    ]
