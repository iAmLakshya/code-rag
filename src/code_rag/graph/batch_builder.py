"""Batched graph builder for high-performance graph operations.

Uses buffered batch processing with UNWIND queries for efficient bulk operations.
This provides significant performance improvements over individual queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from code_rag.graph.client import MemgraphClient
from code_rag.graph.queries import BatchQueries, ProjectQueries
from code_rag.parsing.models import CodeEntity, EntityType, ParsedFile

if TYPE_CHECKING:
    from code_rag.parsing.call_resolution import CallProcessor

logger = logging.getLogger(__name__)


@dataclass
class EntityBuffer:
    """Buffers for batched entity creation."""

    classes: list[dict[str, Any]] = field(default_factory=list)
    functions: list[dict[str, Any]] = field(default_factory=list)
    methods: list[dict[str, Any]] = field(default_factory=list)
    imports: list[dict[str, Any]] = field(default_factory=list)
    files: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RelationshipBuffer:
    """Buffers for batched relationship creation."""

    defines_class: list[dict[str, Any]] = field(default_factory=list)
    defines_function: list[dict[str, Any]] = field(default_factory=list)
    defines_method: list[dict[str, Any]] = field(default_factory=list)
    extends: list[dict[str, Any]] = field(default_factory=list)
    imports: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)


class BatchGraphBuilder:
    """High-performance graph builder using batched operations.

    Uses buffered batch processing similar to the reference implementation.
    Entities and relationships are buffered and flushed in batches using
    UNWIND queries for optimal database performance.

    Usage:
        async with BatchGraphBuilder(client, batch_size=1000) as builder:
            await builder.add_parsed_file(parsed_file)
            # ... add more files ...
            # Flush is automatic on context exit
    """

    def __init__(
        self,
        client: MemgraphClient,
        call_processor: CallProcessor | None = None,
        project_name: str | None = None,
        batch_size: int = 1000,
    ):
        """Initialize batch graph builder.

        Args:
            client: Memgraph client instance.
            call_processor: Optional processor for enhanced call resolution.
            project_name: Project name for qualified name resolution.
            batch_size: Number of items to buffer before auto-flush.
        """
        self.client = client
        self.call_processor = call_processor
        self.project_name = project_name
        self.batch_size = batch_size

        self._entity_buffer = EntityBuffer()
        self._relationship_buffer = RelationshipBuffer()
        self._stats = {"nodes_created": 0, "relationships_created": 0}

        self._current_file_path: str | None = None
        self._current_module_qn: str | None = None
        self._current_language: str | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.warning(f"Exception during batch building: {exc_val}")
        await self.flush_all()
        return False

    async def create_project(self, name: str, path: str) -> None:
        await self.client.execute(
            ProjectQueries.CREATE_PROJECT,
            {"name": name, "path": path},
        )

    def _file_to_module_qn(self, relative_path: str) -> str:
        """Convert file path to module qualified name."""
        path = Path(relative_path)
        parts = list(path.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        base = f"{self.project_name}.{'.'.join(parts)}" if parts else self.project_name
        return base if self.project_name else ".".join(parts)

    async def add_parsed_file(self, parsed_file: ParsedFile) -> None:
        """Add a parsed file to the graph (buffered).

        Args:
            parsed_file: Parsed file with entities.
        """
        file_path = str(parsed_file.file_info.path)

        self._current_file_path = file_path
        self._current_module_qn = self._file_to_module_qn(
            parsed_file.file_info.relative_path
        )
        self._current_language = parsed_file.file_info.language.value

        self._entity_buffer.files.append({
            "path": file_path,
            "name": parsed_file.file_info.path.name,
            "language": parsed_file.file_info.language.value,
            "hash": parsed_file.file_info.content_hash,
            "line_count": parsed_file.file_info.line_count,
            "summary": parsed_file.summary,
        })

        for imp in parsed_file.imports:
            self._entity_buffer.imports.append({
                "name": imp.name,
                "file_path": file_path,
                "alias": imp.alias,
                "source": imp.source,
                "is_external": imp.is_external,
                "line_number": imp.line_number,
            })
            self._relationship_buffer.imports.append({
                "file_path": file_path,
                "import_name": imp.name,
            })

        for entity in parsed_file.entities:
            await self._add_entity(entity, file_path)

        await self._check_auto_flush()

    async def _add_entity(
        self,
        entity: CodeEntity,
        file_path: str,
        parent_class: str | None = None,
    ) -> None:
        """Add a code entity to the buffer."""
        if entity.type == EntityType.CLASS:
            await self._add_class(entity, file_path)
        elif entity.type == EntityType.FUNCTION:
            await self._add_function(entity, file_path)
        elif entity.type == EntityType.METHOD:
            await self._add_method(entity, file_path, parent_class)

    def _build_base_properties(self, entity: CodeEntity, file_path: str) -> dict:
        return {
            "qualified_name": entity.qualified_name,
            "name": entity.name,
            "signature": entity.signature,
            "docstring": entity.docstring,
            "summary": None,
            "start_line": entity.start_line,
            "end_line": entity.end_line,
            "file_path": file_path,
        }

    async def _add_class(self, entity: CodeEntity, file_path: str) -> None:
        props = self._build_base_properties(entity, file_path)
        self._entity_buffer.classes.append(props)

        self._relationship_buffer.defines_class.append({
            "file_path": file_path,
            "class_name": entity.qualified_name,
        })

        for base_class in entity.base_classes:
            self._relationship_buffer.extends.append({
                "child_name": entity.qualified_name,
                "parent_name": base_class,
            })

        for child in entity.children:
            if child.type == EntityType.METHOD:
                await self._add_method(child, file_path, entity.qualified_name)

    async def _add_function(self, entity: CodeEntity, file_path: str) -> None:
        props = self._build_base_properties(entity, file_path)
        props["is_async"] = entity.is_async
        self._entity_buffer.functions.append(props)

        self._relationship_buffer.defines_function.append({
            "file_path": file_path,
            "function_name": entity.qualified_name,
        })

        await self._add_calls_relationships(entity.qualified_name, entity.calls)

    async def _add_method(
        self,
        entity: CodeEntity,
        file_path: str,
        parent_class: str | None,
    ) -> None:
        class_name = parent_class or entity.parent_class

        props = self._build_base_properties(entity, file_path)
        props.update({
            "is_async": entity.is_async,
            "is_static": entity.is_static,
            "is_classmethod": entity.is_classmethod,
            "parent_class": class_name,
        })
        self._entity_buffer.methods.append(props)

        if class_name:
            self._relationship_buffer.defines_method.append({
                "class_name": class_name,
                "method_name": entity.qualified_name,
            })

        await self._add_calls_relationships(
            entity.qualified_name, entity.calls, class_context=class_name
        )

    async def _add_calls_relationships(
        self,
        caller_name: str,
        calls_list: list[str],
        class_context: str | None = None,
    ) -> None:
        """Add CALLS relationships to buffer with optional resolution."""
        for call in calls_list:
            resolved_qn = None

            if self.call_processor and self._current_module_qn:
                try:
                    result = self.call_processor.resolve_call(
                        call_name=call,
                        module_qn=self._current_module_qn,
                        class_context=class_context,
                        language=self._current_language or "python",
                    )
                    if result:
                        _, resolved_qn = result
                        logger.debug(f"CallProcessor resolved: {call} -> {resolved_qn}")
                except Exception as e:
                    logger.debug(f"CallProcessor resolution failed for {call}: {e}")

            callee_name = resolved_qn or call

            self._relationship_buffer.calls.append({
                "caller_name": caller_name,
                "callee_name": callee_name,
            })

    async def _check_auto_flush(self) -> None:
        total_entities = (
            len(self._entity_buffer.classes)
            + len(self._entity_buffer.functions)
            + len(self._entity_buffer.methods)
            + len(self._entity_buffer.imports)
            + len(self._entity_buffer.files)
        )

        if total_entities >= self.batch_size:
            logger.debug(f"Auto-flushing {total_entities} buffered entities")
            await self.flush_entities()

        total_rels = (
            len(self._relationship_buffer.defines_class)
            + len(self._relationship_buffer.defines_function)
            + len(self._relationship_buffer.defines_method)
            + len(self._relationship_buffer.extends)
            + len(self._relationship_buffer.imports)
            + len(self._relationship_buffer.calls)
        )

        if total_rels >= self.batch_size:
            logger.debug(f"Auto-flushing {total_rels} buffered relationships")
            await self.flush_relationships()

    async def flush_entities(self) -> None:
        buf = self._entity_buffer

        if buf.files:
            try:
                await self.client.execute_batch(BatchQueries.BATCH_CREATE_FILE, buf.files)
                self._stats["nodes_created"] += len(buf.files)
            except Exception as e:
                logger.warning(f"Failed to flush files: {e}")
            buf.files.clear()

        if buf.classes:
            try:
                await self.client.execute_batch(BatchQueries.BATCH_CREATE_CLASS, buf.classes)
                self._stats["nodes_created"] += len(buf.classes)
            except Exception as e:
                logger.warning(f"Failed to flush classes: {e}")
            buf.classes.clear()

        if buf.functions:
            try:
                await self.client.execute_batch(BatchQueries.BATCH_CREATE_FUNCTION, buf.functions)
                self._stats["nodes_created"] += len(buf.functions)
            except Exception as e:
                logger.warning(f"Failed to flush functions: {e}")
            buf.functions.clear()

        if buf.methods:
            try:
                await self.client.execute_batch(BatchQueries.BATCH_CREATE_METHOD, buf.methods)
                self._stats["nodes_created"] += len(buf.methods)
            except Exception as e:
                logger.warning(f"Failed to flush methods: {e}")
            buf.methods.clear()

        if buf.imports:
            try:
                await self.client.execute_batch(BatchQueries.BATCH_CREATE_IMPORT, buf.imports)
                self._stats["nodes_created"] += len(buf.imports)
            except Exception as e:
                logger.warning(f"Failed to flush imports: {e}")
            buf.imports.clear()

    async def flush_relationships(self) -> None:
        buf = self._relationship_buffer

        if buf.defines_class:
            try:
                await self.client.execute_batch(
                    BatchQueries.BATCH_CREATE_DEFINES_CLASS, buf.defines_class
                )
                self._stats["relationships_created"] += len(buf.defines_class)
            except Exception as e:
                logger.warning(f"Failed to flush defines_class: {e}")
            buf.defines_class.clear()

        if buf.defines_function:
            try:
                await self.client.execute_batch(
                    BatchQueries.BATCH_CREATE_DEFINES_FUNCTION, buf.defines_function
                )
                self._stats["relationships_created"] += len(buf.defines_function)
            except Exception as e:
                logger.warning(f"Failed to flush defines_function: {e}")
            buf.defines_function.clear()

        if buf.defines_method:
            try:
                await self.client.execute_batch(
                    BatchQueries.BATCH_CREATE_DEFINES_METHOD, buf.defines_method
                )
                self._stats["relationships_created"] += len(buf.defines_method)
            except Exception as e:
                logger.warning(f"Failed to flush defines_method: {e}")
            buf.defines_method.clear()

        if buf.extends:
            try:
                await self.client.execute_batch(
                    BatchQueries.BATCH_CREATE_EXTENDS, buf.extends
                )
                self._stats["relationships_created"] += len(buf.extends)
            except Exception as e:
                logger.warning(f"Failed to flush extends: {e}")
            buf.extends.clear()

        if buf.imports:
            try:
                await self.client.execute_batch(
                    BatchQueries.BATCH_CREATE_IMPORTS, buf.imports
                )
                self._stats["relationships_created"] += len(buf.imports)
            except Exception as e:
                logger.warning(f"Failed to flush imports: {e}")
            buf.imports.clear()

        if buf.calls:
            try:
                results = await self.client.execute_batch(
                    BatchQueries.BATCH_CREATE_CALLS, buf.calls
                )
                created = sum(r.get("created", 0) for r in results) if results else 0
                self._stats["relationships_created"] += created
                if created < len(buf.calls):
                    failed = len(buf.calls) - created
                    logger.debug(f"CALLS: {created} created, {failed} unresolved")
            except Exception as e:
                logger.warning(f"Failed to flush calls: {e}")
            buf.calls.clear()

    async def flush_all(self) -> None:
        logger.info("Flushing all pending graph writes...")
        await self.flush_entities()
        await self.flush_relationships()
        logger.info(
            f"Flush complete: {self._stats['nodes_created']} nodes, "
            f"{self._stats['relationships_created']} relationships"
        )

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)
