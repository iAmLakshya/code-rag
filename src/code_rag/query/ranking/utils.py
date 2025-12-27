from typing import Any

from code_rag.query.ranking.models import RankedResult


def ranked_results_to_search_results(results: list[RankedResult]) -> list[dict[str, Any]]:
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
