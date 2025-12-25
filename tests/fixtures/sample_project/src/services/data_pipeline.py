"""Data processing pipeline with transformation chain pattern.

This module implements a composable data pipeline where:
- Each step transforms data and passes to next step
- Steps can be sync or async
- Pipeline supports branching and merging
- Built-in retry and error handling

Used for ETL operations, data validation, and processing workflows.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Generic,
    TypeVar,
    Awaitable,
    Iterator,
)
from enum import Enum
import asyncio
from collections.abc import AsyncIterator

from code_rag.tests.fixtures.sample_project.src.core.events import (
    EventBus,
    Event,
    EventType,
)


T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


class PipelineStatus(Enum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineContext:
    """Context passed through pipeline stages."""

    pipeline_id: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[Exception] = field(default_factory=list)
    current_stage: str = ""
    stages_completed: list[str] = field(default_factory=list)

    def add_error(self, error: Exception) -> None:
        """Record an error."""
        self.errors.append(error)

    def complete_stage(self, stage_name: str) -> None:
        """Mark a stage as completed."""
        self.stages_completed.append(stage_name)
        self.current_stage = ""

    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0


@dataclass
class StageResult(Generic[T]):
    """Result from a pipeline stage."""

    data: T
    success: bool = True
    error: Exception | None = None
    duration_ms: float = 0
    records_processed: int = 0
    records_failed: int = 0


class PipelineStage(ABC, Generic[T, U]):
    """Abstract base class for pipeline stages.

    Each stage takes input of type T and produces output of type U.
    Stages are composable using the >> operator.
    """

    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def process(
        self,
        data: T,
        context: PipelineContext,
    ) -> StageResult[U]:
        """Process input data and return result."""
        pass

    def __rshift__(self, other: "PipelineStage[U, V]") -> "ChainedStage[T, V]":
        """Chain stages using >> operator."""
        return ChainedStage(self, other)


class ChainedStage(PipelineStage[T, V]):
    """Two stages chained together."""

    def __init__(
        self,
        first: PipelineStage[T, U],
        second: PipelineStage[U, V],
    ):
        super().__init__(f"{first.name} >> {second.name}")
        self.first = first
        self.second = second

    async def process(
        self,
        data: T,
        context: PipelineContext,
    ) -> StageResult[V]:
        """Process through both stages."""
        # Run first stage
        result1 = await self.first.process(data, context)
        if not result1.success:
            return StageResult(
                data=None,  # type: ignore
                success=False,
                error=result1.error,
            )

        # Run second stage with first's output
        result2 = await self.second.process(result1.data, context)
        return StageResult(
            data=result2.data,
            success=result2.success,
            error=result2.error,
            duration_ms=result1.duration_ms + result2.duration_ms,
            records_processed=result2.records_processed,
            records_failed=result1.records_failed + result2.records_failed,
        )


class MapStage(PipelineStage[list[T], list[U]]):
    """Apply a function to each item in a list.

    Supports both sync and async mapping functions.
    Failed items are collected but don't stop processing.
    """

    def __init__(
        self,
        func: Callable[[T], U | Awaitable[U]],
        name: str | None = None,
        continue_on_error: bool = True,
    ):
        super().__init__(name or f"Map({func.__name__})")
        self.func = func
        self.continue_on_error = continue_on_error

    async def process(
        self,
        data: list[T],
        context: PipelineContext,
    ) -> StageResult[list[U]]:
        """Apply function to each item."""
        import time

        start = time.perf_counter()
        results: list[U] = []
        failed = 0

        for item in data:
            try:
                result = self.func(item)
                if asyncio.iscoroutine(result):
                    result = await result
                results.append(result)
            except Exception as e:
                failed += 1
                context.add_error(e)
                if not self.continue_on_error:
                    return StageResult(
                        data=results,
                        success=False,
                        error=e,
                        records_processed=len(results),
                        records_failed=failed,
                    )

        duration = (time.perf_counter() - start) * 1000
        return StageResult(
            data=results,
            success=True,
            duration_ms=duration,
            records_processed=len(results),
            records_failed=failed,
        )


class FilterStage(PipelineStage[list[T], list[T]]):
    """Filter items based on a predicate."""

    def __init__(
        self,
        predicate: Callable[[T], bool | Awaitable[bool]],
        name: str | None = None,
    ):
        super().__init__(name or f"Filter({predicate.__name__})")
        self.predicate = predicate

    async def process(
        self,
        data: list[T],
        context: PipelineContext,
    ) -> StageResult[list[T]]:
        """Filter items matching predicate."""
        import time

        start = time.perf_counter()
        results: list[T] = []

        for item in data:
            result = self.predicate(item)
            if asyncio.iscoroutine(result):
                result = await result
            if result:
                results.append(item)

        duration = (time.perf_counter() - start) * 1000
        return StageResult(
            data=results,
            success=True,
            duration_ms=duration,
            records_processed=len(data),
        )


class BatchStage(PipelineStage[list[T], list[list[T]]]):
    """Split items into batches for parallel processing."""

    def __init__(self, batch_size: int, name: str | None = None):
        super().__init__(name or f"Batch({batch_size})")
        self.batch_size = batch_size

    async def process(
        self,
        data: list[T],
        context: PipelineContext,
    ) -> StageResult[list[list[T]]]:
        """Split into batches."""
        batches = [
            data[i:i + self.batch_size]
            for i in range(0, len(data), self.batch_size)
        ]
        return StageResult(
            data=batches,
            success=True,
            records_processed=len(data),
        )


class FlattenStage(PipelineStage[list[list[T]], list[T]]):
    """Flatten nested lists into a single list."""

    def __init__(self, name: str | None = None):
        super().__init__(name or "Flatten")

    async def process(
        self,
        data: list[list[T]],
        context: PipelineContext,
    ) -> StageResult[list[T]]:
        """Flatten nested lists."""
        flat = [item for batch in data for item in batch]
        return StageResult(
            data=flat,
            success=True,
            records_processed=len(flat),
        )


class AggregateStage(PipelineStage[list[T], U]):
    """Aggregate items into a single result."""

    def __init__(
        self,
        aggregator: Callable[[list[T]], U | Awaitable[U]],
        name: str | None = None,
    ):
        super().__init__(name or f"Aggregate({aggregator.__name__})")
        self.aggregator = aggregator

    async def process(
        self,
        data: list[T],
        context: PipelineContext,
    ) -> StageResult[U]:
        """Aggregate all items."""
        result = self.aggregator(data)
        if asyncio.iscoroutine(result):
            result = await result
        return StageResult(
            data=result,
            success=True,
            records_processed=len(data),
        )


class ValidateStage(PipelineStage[T, T]):
    """Validate data against rules."""

    def __init__(
        self,
        validator: Callable[[T], bool | str | Awaitable[bool | str]],
        name: str | None = None,
    ):
        super().__init__(name or "Validate")
        self.validator = validator

    async def process(
        self,
        data: T,
        context: PipelineContext,
    ) -> StageResult[T]:
        """Validate data."""
        result = self.validator(data)
        if asyncio.iscoroutine(result):
            result = await result

        if result is True:
            return StageResult(data=data, success=True)
        elif result is False:
            return StageResult(
                data=data,
                success=False,
                error=ValueError("Validation failed"),
            )
        else:
            # String result is an error message
            return StageResult(
                data=data,
                success=False,
                error=ValueError(result),
            )


class BranchStage(PipelineStage[T, dict[str, Any]]):
    """Branch pipeline into multiple parallel paths.

    Each branch processes the same input independently.
    Results are collected into a dictionary.
    """

    def __init__(
        self,
        branches: dict[str, PipelineStage[T, Any]],
        name: str | None = None,
    ):
        super().__init__(name or "Branch")
        self.branches = branches

    async def process(
        self,
        data: T,
        context: PipelineContext,
    ) -> StageResult[dict[str, Any]]:
        """Execute all branches in parallel."""
        tasks = {
            name: asyncio.create_task(stage.process(data, context))
            for name, stage in self.branches.items()
        }

        results = {}
        failed = False
        errors = []

        for name, task in tasks.items():
            result = await task
            results[name] = result.data
            if not result.success:
                failed = True
                errors.append(result.error)

        return StageResult(
            data=results,
            success=not failed,
            error=errors[0] if errors else None,
        )


class Pipeline(Generic[T, U]):
    """Composable data processing pipeline.

    Example:
        pipeline = (
            Pipeline[list[dict], list[User]]("user_import")
            .add_stage(ValidateStage(validate_user_data))
            .add_stage(MapStage(transform_to_user))
            .add_stage(FilterStage(lambda u: u.is_active))
            .add_stage(BatchStage(100))
            .add_stage(MapStage(save_user_batch))
            .add_stage(FlattenStage())
        )

        result = await pipeline.execute(raw_data)
    """

    def __init__(self, name: str):
        self.name = name
        self._stages: list[PipelineStage] = []
        self._hooks: dict[str, list[Callable]] = {
            "before_stage": [],
            "after_stage": [],
            "on_error": [],
            "on_complete": [],
        }

    def add_stage(self, stage: PipelineStage) -> "Pipeline":
        """Add a stage to the pipeline."""
        self._stages.append(stage)
        return self

    def add_hook(
        self,
        event: str,
        callback: Callable,
    ) -> "Pipeline":
        """Add a hook for pipeline events."""
        if event in self._hooks:
            self._hooks[event].append(callback)
        return self

    async def execute(self, data: T) -> StageResult[U]:
        """Execute the pipeline."""
        import time
        import uuid

        context = PipelineContext(
            pipeline_id=str(uuid.uuid4()),
            metadata={"pipeline_name": self.name},
        )

        current_data: Any = data
        total_duration = 0

        await EventBus.get_instance().publish(
            Event(
                type=EventType.ITEM_CREATED,
                payload={
                    "pipeline_id": context.pipeline_id,
                    "pipeline_name": self.name,
                    "stage_count": len(self._stages),
                },
                source="data_pipeline",
            )
        )

        for stage in self._stages:
            context.current_stage = stage.name

            # Before hooks
            for hook in self._hooks["before_stage"]:
                await self._call_hook(hook, stage.name, current_data, context)

            try:
                result = await stage.process(current_data, context)
                total_duration += result.duration_ms

                if not result.success:
                    # Error hooks
                    for hook in self._hooks["on_error"]:
                        await self._call_hook(
                            hook, stage.name, result.error, context
                        )

                    return StageResult(
                        data=current_data,
                        success=False,
                        error=result.error,
                        duration_ms=total_duration,
                    )

                current_data = result.data
                context.complete_stage(stage.name)

                # After hooks
                for hook in self._hooks["after_stage"]:
                    await self._call_hook(hook, stage.name, current_data, context)

            except Exception as e:
                context.add_error(e)
                for hook in self._hooks["on_error"]:
                    await self._call_hook(hook, stage.name, e, context)

                return StageResult(
                    data=current_data,
                    success=False,
                    error=e,
                    duration_ms=total_duration,
                )

        # Complete hooks
        for hook in self._hooks["on_complete"]:
            await self._call_hook(hook, current_data, context)

        await EventBus.get_instance().publish(
            Event(
                type=EventType.ITEM_UPDATED,
                payload={
                    "pipeline_id": context.pipeline_id,
                    "status": "completed",
                    "duration_ms": total_duration,
                    "stages_completed": context.stages_completed,
                },
                source="data_pipeline",
            )
        )

        return StageResult(
            data=current_data,
            success=True,
            duration_ms=total_duration,
        )

    async def _call_hook(self, hook: Callable, *args) -> None:
        """Call a hook function."""
        result = hook(*args)
        if asyncio.iscoroutine(result):
            await result


# Convenience functions for creating common pipelines


def create_etl_pipeline(
    extract: Callable[[], Awaitable[list[dict]]],
    transform: Callable[[dict], Any],
    load: Callable[[list[Any]], Awaitable[int]],
) -> Pipeline:
    """Create a standard ETL pipeline."""

    class ExtractStage(PipelineStage[None, list[dict]]):
        async def process(self, data, context):
            result = await extract()
            return StageResult(data=result, records_processed=len(result))

    class LoadStage(PipelineStage[list[Any], int]):
        async def process(self, data, context):
            count = await load(data)
            return StageResult(data=count, records_processed=count)

    return (
        Pipeline("etl")
        .add_stage(ExtractStage("Extract"))
        .add_stage(MapStage(transform, "Transform"))
        .add_stage(LoadStage("Load"))
    )
