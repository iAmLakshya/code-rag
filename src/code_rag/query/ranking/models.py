from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from code_rag.core.types import ResultSource
from code_rag.query.query_planner import QueryIntent

DEFAULT_GRAPH_WEIGHT = 0.5
DEFAULT_VECTOR_WEIGHT = 0.5
DEFAULT_CENTRALITY_WEIGHT = 0.2
DEFAULT_CONTEXT_WEIGHT = 0.1

MAX_RESULTS_PER_FILE = 5
MAX_TOTAL_RESULTS = 50


class RankingSignal(Enum):
    GRAPH_MATCH = "graph_match"
    VECTOR_SIMILARITY = "vector_similarity"
    CENTRALITY = "centrality"
    QUERY_ENTITY_MATCH = "query_entity_match"
    RELATIONSHIP_RELEVANCE = "relationship_relevance"
    CODE_QUALITY = "code_quality"
    CONTEXT_RICHNESS = "context_richness"


@dataclass
class RankedResult:
    file_path: str
    entity_name: str
    entity_type: str
    qualified_name: str | None = None

    content: str | None = None
    summary: str | None = None
    signature: str | None = None
    docstring: str | None = None

    start_line: int | None = None
    end_line: int | None = None

    source: str = ResultSource.HYBRID.value
    graph_node_id: str | None = None

    final_score: float = 0.0
    signal_scores: dict[str, float] = field(default_factory=dict)

    callers: list[str] = field(default_factory=list)
    callees: list[str] = field(default_factory=list)
    depth_from_query: int | None = None
    relationship_path: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def get_key(self) -> str:
        return f"{self.file_path}:{self.entity_name}:{self.start_line}"


@dataclass
class RankingConfig:
    graph_weight: float = DEFAULT_GRAPH_WEIGHT
    vector_weight: float = DEFAULT_VECTOR_WEIGHT

    centrality_weight: float = DEFAULT_CENTRALITY_WEIGHT
    context_weight: float = DEFAULT_CONTEXT_WEIGHT
    entity_match_bonus: float = 0.3
    relationship_bonus: float = 0.15

    query_type_adjustments: dict[QueryIntent, dict[str, float]] = field(default_factory=dict)

    max_per_file: int = MAX_RESULTS_PER_FILE
    max_total: int = MAX_TOTAL_RESULTS

    def __post_init__(self):
        if not self.query_type_adjustments:
            self.query_type_adjustments = {
                QueryIntent.FIND_CALLERS: {"graph_weight": 0.8, "vector_weight": 0.2},
                QueryIntent.FIND_CALLEES: {"graph_weight": 0.8, "vector_weight": 0.2},
                QueryIntent.FIND_CALL_CHAIN: {"graph_weight": 0.9, "vector_weight": 0.1},
                QueryIntent.FIND_HIERARCHY: {"graph_weight": 0.85, "vector_weight": 0.15},
                QueryIntent.FIND_USAGES: {"graph_weight": 0.7, "vector_weight": 0.3},
                QueryIntent.FIND_DEPENDENCIES: {"graph_weight": 0.75, "vector_weight": 0.25},
                QueryIntent.LOCATE_ENTITY: {"graph_weight": 0.6, "vector_weight": 0.4},
                QueryIntent.LOCATE_FILE: {"graph_weight": 0.5, "vector_weight": 0.5},
                QueryIntent.EXPLAIN_IMPLEMENTATION: {"graph_weight": 0.5, "vector_weight": 0.5},
                QueryIntent.EXPLAIN_RELATIONSHIP: {"graph_weight": 0.6, "vector_weight": 0.4},
                QueryIntent.EXPLAIN_DATA_FLOW: {"graph_weight": 0.65, "vector_weight": 0.35},
                QueryIntent.FIND_SIMILAR: {"graph_weight": 0.2, "vector_weight": 0.8},
                QueryIntent.SEARCH_FUNCTIONALITY: {"graph_weight": 0.3, "vector_weight": 0.7},
                QueryIntent.SEARCH_PATTERN: {"graph_weight": 0.25, "vector_weight": 0.75},
            }
