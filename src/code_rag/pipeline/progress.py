"""Progress tracking for the indexing pipeline."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Callable

from code_rag.core.types import PipelineStage

logger = logging.getLogger(__name__)

STAGE_WEIGHT_SCANNING = 5
STAGE_WEIGHT_PARSING = 15
STAGE_WEIGHT_BUILDING_GRAPH = 20
STAGE_WEIGHT_SUMMARIZING = 30
STAGE_WEIGHT_EMBEDDING = 30


@dataclass
class StageProgress:
    """Progress for a single stage."""

    stage: PipelineStage
    current: int = 0
    total: int = 0
    message: str = ""

    @property
    def percentage(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100


@dataclass
class PipelineProgress:
    """Overall progress of the indexing pipeline."""

    current_stage: PipelineStage = PipelineStage.SCANNING
    stages: dict[PipelineStage, StageProgress] = field(default_factory=dict)
    start_time: datetime | None = None
    end_time: datetime | None = None
    error_message: str | None = None

    files_scanned: int = 0
    files_parsed: int = 0
    entities_found: int = 0
    graph_nodes_created: int = 0
    summaries_generated: int = 0
    chunks_embedded: int = 0

    @property
    def is_running(self) -> bool:
        return self.start_time is not None and self.current_stage not in (
            PipelineStage.COMPLETED,
            PipelineStage.FAILED,
        )

    @property
    def is_complete(self) -> bool:
        return self.current_stage == PipelineStage.COMPLETED

    @property
    def has_error(self) -> bool:
        return self.current_stage == PipelineStage.FAILED

    @property
    def elapsed_time(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    @property
    def overall_percentage(self) -> float:
        stage_weights = {
            PipelineStage.SCANNING: STAGE_WEIGHT_SCANNING,
            PipelineStage.PARSING: STAGE_WEIGHT_PARSING,
            PipelineStage.GRAPH_BUILDING: STAGE_WEIGHT_BUILDING_GRAPH,
            PipelineStage.SUMMARIZING: STAGE_WEIGHT_SUMMARIZING,
            PipelineStage.EMBEDDING: STAGE_WEIGHT_EMBEDDING,
        }

        completed_weight = 0
        for stage, weight in stage_weights.items():
            if stage in self.stages:
                progress = self.stages[stage]
                if progress.total > 0:
                    stage_progress = progress.current / progress.total
                    completed_weight += weight * stage_progress
                elif self._is_stage_complete(stage):
                    completed_weight += weight

        return min(completed_weight, 100)

    def _is_stage_complete(self, stage: PipelineStage) -> bool:
        stage_order = [
            PipelineStage.SCANNING,
            PipelineStage.PARSING,
            PipelineStage.GRAPH_BUILDING,
            PipelineStage.SUMMARIZING,
            PipelineStage.EMBEDDING,
            PipelineStage.COMPLETED,
            PipelineStage.FAILED,
        ]
        current_idx = stage_order.index(self.current_stage)
        stage_idx = stage_order.index(stage)
        return stage_idx < current_idx


class ProgressTracker:
    """Tracks and reports pipeline progress with thread-safety."""

    def __init__(self):
        self._progress = PipelineProgress()
        self._callbacks: list[Callable[[PipelineProgress], None]] = []
        self._lock = Lock()

    @property
    def progress(self) -> PipelineProgress:
        with self._lock:
            return self._progress

    def add_callback(self, callback: Callable[[PipelineProgress], None]) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[PipelineProgress], None]) -> None:
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def _notify(self) -> None:
        with self._lock:
            callbacks = self._callbacks.copy()
            progress = self._progress

        for callback in callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}", exc_info=True)

    def start(self) -> None:
        with self._lock:
            self._progress = PipelineProgress(
                current_stage=PipelineStage.SCANNING,
                start_time=datetime.now(),
            )
        self._notify()

    def set_stage(self, stage: PipelineStage, total: int = 0, message: str = "") -> None:
        with self._lock:
            self._progress.current_stage = stage
            self._progress.stages[stage] = StageProgress(
                stage=stage,
                current=0,
                total=total,
                message=message,
            )
        self._notify()

    def update_stage(
        self,
        current: int,
        total: int | None = None,
        message: str | None = None,
    ) -> None:
        with self._lock:
            stage = self._progress.current_stage
            if stage in self._progress.stages:
                progress = self._progress.stages[stage]
                progress.current = current
                if total is not None:
                    progress.total = total
                if message is not None:
                    progress.message = message
        self._notify()

    def increment_stage(self, message: str | None = None) -> None:
        with self._lock:
            stage = self._progress.current_stage
            if stage in self._progress.stages:
                progress = self._progress.stages[stage]
                progress.current += 1
                if message is not None:
                    progress.message = message
        self._notify()

    def update_stats(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._progress, key):
                    setattr(self._progress, key, value)
        self._notify()

    def complete(self) -> None:
        with self._lock:
            self._progress.current_stage = PipelineStage.COMPLETED
            self._progress.end_time = datetime.now()
        self._notify()

    def error(self, message: str) -> None:
        with self._lock:
            self._progress.current_stage = PipelineStage.FAILED
            self._progress.error_message = message
            self._progress.end_time = datetime.now()
        self._notify()

    def reset(self) -> None:
        with self._lock:
            self._progress = PipelineProgress()
        self._notify()
