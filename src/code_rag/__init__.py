"""Code RAG - A hybrid RAG system for code repositories."""

__version__ = "0.1.0"

from code_rag.config import Settings, get_settings
from code_rag.pipeline.orchestrator import PipelineOrchestrator, run_indexing
from code_rag.query import QueryEngine, QueryResult

__all__ = [
    "get_settings",
    "PipelineOrchestrator",
    "QueryEngine",
    "QueryResult",
    "run_indexing",
    "Settings",
]
