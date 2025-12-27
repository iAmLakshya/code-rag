from code_rag.core.types import ResultSource
from code_rag.query.ranking.models import RankedResult, RankingConfig, RankingSignal


class ResultScorer:
    def __init__(self, config: RankingConfig):
        self.config = config

    def score_graph_result(
        self,
        result: RankedResult,
        weights: dict[str, float],
        centrality_scores: dict[str, dict[str, int]],
        query_entities: set[str],
        is_primary: bool = False,
        is_caller: bool = False,
        is_callee: bool = False,
    ) -> None:
        signals = {}

        base_score = 1.0
        if is_primary:
            base_score = 1.0
        elif is_caller or is_callee:
            depth = result.depth_from_query or 1
            base_score = max(0.3, 1.0 - (depth - 1) * 0.2)

        signals[RankingSignal.GRAPH_MATCH.value] = base_score

        entity_match = 0.0
        if result.entity_name.lower() in query_entities:
            entity_match = 1.0
        elif any(qe in result.entity_name.lower() for qe in query_entities):
            entity_match = 0.5
        signals[RankingSignal.QUERY_ENTITY_MATCH.value] = entity_match

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

        centrality_score = 0.0
        entity_key = result.qualified_name or result.entity_name
        if entity_key in centrality_scores:
            scores = centrality_scores[entity_key]
            total_degree = scores.get("total_degree", 0)
            centrality_score = min(1.0, total_degree / 50)
        signals[RankingSignal.CENTRALITY.value] = centrality_score

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

    def score_vector_result(
        self,
        result: RankedResult,
        vector_score: float,
        weights: dict[str, float],
        centrality_scores: dict[str, dict[str, int]],
        query_entities: set[str],
    ) -> None:
        signals = {}

        signals[RankingSignal.VECTOR_SIMILARITY.value] = vector_score

        entity_match = 0.0
        if result.entity_name.lower() in query_entities:
            entity_match = 1.0
        elif any(qe in result.entity_name.lower() for qe in query_entities):
            entity_match = 0.5
        signals[RankingSignal.QUERY_ENTITY_MATCH.value] = entity_match

        centrality_score = 0.0
        entity_key = result.qualified_name or result.entity_name
        if entity_key in centrality_scores:
            scores = centrality_scores[entity_key]
            total_degree = scores.get("total_degree", 0)
            centrality_score = min(1.0, total_degree / 50)
        signals[RankingSignal.CENTRALITY.value] = centrality_score

        quality_score = 0.0
        if result.content:
            content_len = len(result.content)
            if 100 < content_len < 2000:
                quality_score = 0.8
            elif 50 < content_len < 3000:
                quality_score = 0.5
            else:
                quality_score = 0.3
        signals[RankingSignal.CODE_QUALITY.value] = quality_score

        final_score = (
            signals[RankingSignal.VECTOR_SIMILARITY.value] * weights["vector_weight"]
            + signals.get(RankingSignal.QUERY_ENTITY_MATCH.value, 0) * self.config.entity_match_bonus
            + signals.get(RankingSignal.CENTRALITY.value, 0) * weights["centrality_weight"]
            + signals.get(RankingSignal.CODE_QUALITY.value, 0) * 0.1
        )

        result.final_score = final_score
        result.signal_scores = signals
        result.source = ResultSource.VECTOR.value
