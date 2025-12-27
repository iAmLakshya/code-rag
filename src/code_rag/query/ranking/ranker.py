import logging
from typing import Any

from code_rag.core.types import ResultSource
from code_rag.query.graph_reasoning import GraphContext, GraphNode
from code_rag.query.query_planner import QueryIntent, QueryPlan
from code_rag.query.ranking.models import RankedResult, RankingConfig
from code_rag.query.ranking.scorer import ResultScorer

logger = logging.getLogger(__name__)


class HybridRanker:
    def __init__(self, config: RankingConfig | None = None):
        self.config = config or RankingConfig()
        self.scorer = ResultScorer(self.config)

    def rank_results(
        self,
        plan: QueryPlan,
        graph_context: GraphContext,
        vector_results: list[dict[str, Any]],
        centrality_scores: dict[str, dict[str, int]] | None = None,
    ) -> list[RankedResult]:
        logger.debug(f"Ranking results: graph={len(self._count_graph_entities(graph_context))}, vector={len(vector_results)}")

        weights = self._get_adjusted_weights(plan.primary_intent)

        results_map: dict[str, RankedResult] = {}

        self._process_graph_results(
            results_map,
            graph_context,
            plan,
            weights,
            centrality_scores or {},
        )

        self._process_vector_results(
            results_map,
            vector_results,
            plan,
            weights,
            centrality_scores or {},
        )

        results = list(results_map.values())
        results.sort(key=lambda r: r.final_score, reverse=True)

        deduplicated = self._deduplicate_and_limit(results)

        logger.debug(f"Ranking complete: {len(deduplicated)} results after deduplication")

        return deduplicated

    def _get_adjusted_weights(self, intent: QueryIntent) -> dict[str, float]:
        base_weights = {
            "graph_weight": self.config.graph_weight,
            "vector_weight": self.config.vector_weight,
            "centrality_weight": self.config.centrality_weight,
            "context_weight": self.config.context_weight,
        }

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
        query_entities = {e.name.lower() for e in plan.entities}

        for node in graph_context.primary_entities:
            result = self._graph_node_to_result(node)
            self.scorer.score_graph_result(
                result,
                weights,
                centrality_scores,
                query_entities,
                is_primary=True,
            )
            self._merge_into_map(results_map, result)

        for node in graph_context.callers:
            result = self._graph_node_to_result(node)
            result.relationship_path = "caller"
            result.depth_from_query = node.metadata.get("depth", 1) if node.metadata else 1
            self.scorer.score_graph_result(
                result,
                weights,
                centrality_scores,
                query_entities,
                is_caller=True,
            )
            self._merge_into_map(results_map, result)

        for node in graph_context.callees:
            result = self._graph_node_to_result(node)
            result.relationship_path = "callee"
            result.depth_from_query = node.metadata.get("depth", 1) if node.metadata else 1
            self.scorer.score_graph_result(
                result,
                weights,
                centrality_scores,
                query_entities,
                is_callee=True,
            )
            self._merge_into_map(results_map, result)

        for node in graph_context.methods:
            result = self._graph_node_to_result(node)
            result.relationship_path = "method"
            self.scorer.score_graph_result(
                result,
                weights,
                centrality_scores,
                query_entities,
            )
            self._merge_into_map(results_map, result)

        for node in graph_context.parent_classes:
            result = self._graph_node_to_result(node)
            result.relationship_path = "parent_class"
            self.scorer.score_graph_result(
                result,
                weights,
                centrality_scores,
                query_entities,
            )
            self._merge_into_map(results_map, result)

        for node in graph_context.child_classes:
            result = self._graph_node_to_result(node)
            result.relationship_path = "child_class"
            self.scorer.score_graph_result(
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
        query_entities = {e.name.lower() for e in plan.entities}

        for vr in vector_results:
            result = self._vector_result_to_ranked(vr)
            self.scorer.score_vector_result(
                result,
                vr.get("score", 0.0),
                weights,
                centrality_scores,
                query_entities,
            )
            self._merge_into_map(results_map, result)

    def _merge_into_map(
        self,
        results_map: dict[str, RankedResult],
        result: RankedResult,
    ) -> None:
        key = result.get_key()

        if key not in results_map:
            results_map[key] = result
        else:
            existing = results_map[key]

            combined_score = (existing.final_score + result.final_score) / 2
            combined_score *= 1.1

            if not existing.content and result.content:
                existing.content = result.content
            if not existing.summary and result.summary:
                existing.summary = result.summary
            if not existing.signature and result.signature:
                existing.signature = result.signature
            if not existing.docstring and result.docstring:
                existing.docstring = result.docstring

            for signal, score in result.signal_scores.items():
                if signal in existing.signal_scores:
                    existing.signal_scores[signal] = max(existing.signal_scores[signal], score)
                else:
                    existing.signal_scores[signal] = score

            existing.final_score = combined_score
            existing.source = ResultSource.HYBRID.value

    def _deduplicate_and_limit(self, results: list[RankedResult]) -> list[RankedResult]:
        seen_keys = set()
        file_counts: dict[str, int] = {}
        deduplicated = []

        for result in results:
            key = result.get_key()

            if key in seen_keys:
                continue

            file_count = file_counts.get(result.file_path, 0)
            if file_count >= self.config.max_per_file:
                continue

            seen_keys.add(key)
            file_counts[result.file_path] = file_count + 1
            deduplicated.append(result)

            if len(deduplicated) >= self.config.max_total:
                break

        return deduplicated

    def _graph_node_to_result(self, node: GraphNode) -> RankedResult:
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
        entities = []
        entities.extend(graph_context.primary_entities)
        entities.extend(graph_context.callers)
        entities.extend(graph_context.callees)
        entities.extend(graph_context.methods)
        entities.extend(graph_context.parent_classes)
        entities.extend(graph_context.child_classes)
        return entities
