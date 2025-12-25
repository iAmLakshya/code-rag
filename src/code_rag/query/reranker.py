"""Result fusion and reranking for hybrid search."""

from dataclasses import dataclass
from typing import Any

from code_rag.core.types import ResultSource

DEFAULT_GRAPH_WEIGHT = 0.4
DEFAULT_VECTOR_WEIGHT = 0.6
DEFAULT_MAX_RESULTS_PER_FILE = 3


@dataclass
class SearchResult:
    """Unified search result from any source."""

    source: str
    score: float
    file_path: str
    entity_type: str
    entity_name: str
    content: str | None = None
    summary: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    graph_node_id: str | None = None
    metadata: dict[str, Any] | None = None

    def get_key(self) -> str:
        """Generate unique key for deduplication."""
        return f"{self.file_path}:{self.entity_name}:{self.start_line}"


def normalize_scores(results: list[SearchResult]) -> list[SearchResult]:
    """Normalize scores to 0-1 range.

    Args:
        results: Results to normalize.

    Returns:
        New list with normalized scores.
    """
    if not results:
        return results

    max_score = max(r.score for r in results)
    min_score = min(r.score for r in results)
    score_range = max_score - min_score

    if score_range == 0:
        return [
            SearchResult(
                source=r.source,
                score=1.0,
                file_path=r.file_path,
                entity_type=r.entity_type,
                entity_name=r.entity_name,
                content=r.content,
                summary=r.summary,
                start_line=r.start_line,
                end_line=r.end_line,
                graph_node_id=r.graph_node_id,
                metadata=r.metadata,
            )
            for r in results
        ]

    return [
        SearchResult(
            source=r.source,
            score=(r.score - min_score) / score_range,
            file_path=r.file_path,
            entity_type=r.entity_type,
            entity_name=r.entity_name,
            content=r.content,
            summary=r.summary,
            start_line=r.start_line,
            end_line=r.end_line,
            graph_node_id=r.graph_node_id,
            metadata=r.metadata,
        )
        for r in results
    ]


class ResultReranker:
    """Fuses and reranks results from multiple sources."""

    def __init__(
        self,
        graph_weight: float = DEFAULT_GRAPH_WEIGHT,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    ):
        """Initialize the reranker.

        Args:
            graph_weight: Weight for graph search results.
            vector_weight: Weight for vector search results.
        """
        self.graph_weight = graph_weight
        self.vector_weight = vector_weight

    def fuse_results(
        self,
        graph_results: list[dict],
        vector_results: list[dict],
    ) -> list[SearchResult]:
        """Fuse results from graph and vector search.

        Args:
            graph_results: Results from graph search.
            vector_results: Results from vector search.

        Returns:
            Fused and ranked results.
        """
        results_map: dict[str, SearchResult] = {}

        for r in graph_results:
            result = self._create_graph_result(r)
            results_map[result.get_key()] = result

        for r in vector_results:
            result = self._create_vector_result(r)
            key = result.get_key()

            if key in results_map:
                existing = results_map[key]
                results_map[key] = SearchResult(
                    source=ResultSource.HYBRID.value,
                    score=existing.score + result.score,
                    file_path=existing.file_path,
                    entity_type=existing.entity_type,
                    entity_name=existing.entity_name,
                    content=result.content or existing.content,
                    summary=existing.summary or result.summary,
                    start_line=existing.start_line,
                    end_line=existing.end_line,
                    graph_node_id=existing.graph_node_id,
                    metadata=existing.metadata,
                )
            else:
                results_map[key] = result

        results = list(results_map.values())
        results.sort(key=lambda r: r.score, reverse=True)

        return results

    def deduplicate(
        self,
        results: list[SearchResult],
        max_per_file: int = DEFAULT_MAX_RESULTS_PER_FILE,
    ) -> list[SearchResult]:
        """Remove duplicate and limit results per file.

        Args:
            results: Results to deduplicate.
            max_per_file: Maximum results per file.

        Returns:
            Deduplicated results.
        """
        seen_keys = set()
        file_counts: dict[str, int] = {}
        deduplicated = []

        for result in results:
            key = result.get_key()

            if key in seen_keys:
                continue

            file_count = file_counts.get(result.file_path, 0)
            if file_count >= max_per_file:
                continue

            seen_keys.add(key)
            file_counts[result.file_path] = file_count + 1
            deduplicated.append(result)

        return deduplicated

    def _create_graph_result(self, r: dict) -> SearchResult:
        """Create SearchResult from graph result dictionary."""
        return SearchResult(
            source=ResultSource.GRAPH.value,
            score=self.graph_weight,
            file_path=r.get("file_path", ""),
            entity_type=r.get("type", r.get("entity_type", "")),
            entity_name=r.get("name", r.get("entity_name", "")),
            summary=r.get("summary"),
            start_line=r.get("start_line"),
            end_line=r.get("end_line"),
            graph_node_id=r.get("qualified_name"),
        )

    def _create_vector_result(self, r: dict) -> SearchResult:
        """Create SearchResult from vector result dictionary."""
        return SearchResult(
            source=ResultSource.VECTOR.value,
            score=r.get("score", 0) * self.vector_weight,
            file_path=r.get("file_path", ""),
            entity_type=r.get("entity_type", ""),
            entity_name=r.get("entity_name", ""),
            content=r.get("content"),
            summary=r.get("summary"),
            start_line=r.get("start_line"),
            end_line=r.get("end_line"),
            graph_node_id=r.get("graph_node_id"),
        )
