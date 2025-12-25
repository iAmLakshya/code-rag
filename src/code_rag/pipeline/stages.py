"""Pipeline stage definitions - re-exports from core.types."""

from code_rag.core.types import PipelineStage
from code_rag.pipeline.progress import StageProgress

__all__ = ["PipelineStage", "StageProgress"]
