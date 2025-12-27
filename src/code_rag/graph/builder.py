"""Graph builder for constructing knowledge graph from parsed code."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from code_rag.graph.client import MemgraphClient
from code_rag.graph.queries import EntityQueries, FileQueries, ProjectQueries, RelationshipQueries
from code_rag.parsing.models import CodeEntity, EntityType, ParsedFile

if TYPE_CHECKING:
    from code_rag.parsing.call_resolution import CallProcessor

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds the knowledge graph from parsed code files.

    Optionally uses CallProcessor for enhanced call resolution with:
    - Inheritance chain resolution
    - Import-based resolution
    - Type inference for method calls
    """

    def __init__(
        self,
        client: MemgraphClient,
        call_processor: CallProcessor | None = None,
        project_name: str | None = None,
    ):
        """Initialize graph builder.

        Args:
            client: Memgraph client instance.
            call_processor: Optional processor for enhanced call resolution.
            project_name: Project name for qualified name resolution.
        """
        self.client = client
        self.call_processor = call_processor
        self.project_name = project_name
        self._current_file_path: str | None = None
        self._current_module_qn: str | None = None

    async def create_project(self, name: str, path: str) -> None:
        """Create or update a project node.

        Args:
            name: Project name.
            path: Project root path.
        """
        await self.client.execute(
            ProjectQueries.CREATE_PROJECT,
            {"name": name, "path": path},
        )

    async def file_needs_update(self, path: str, content_hash: str) -> bool:
        """Check if a file needs to be reindexed.

        Args:
            path: File path.
            content_hash: Current content hash.

        Returns:
            True if file needs update.
        """
        result = await self.client.execute(
            FileQueries.GET_FILE_BY_HASH,
            {"path": path, "hash": content_hash},
        )
        return len(result) == 0

    async def delete_file_entities(self, path: str) -> None:
        """Delete all entities associated with a file.

        Args:
            path: File path.
        """
        await self.client.execute(
            FileQueries.DELETE_FILE_ENTITIES,
            {"path": path},
        )

    async def build_from_parsed_file(self, parsed_file: ParsedFile) -> None:
        """Build graph nodes and relationships from a parsed file.

        Args:
            parsed_file: Parsed file with entities.
        """
        file_path = str(parsed_file.file_info.path)

        # Set context for call resolution
        self._current_file_path = file_path
        self._current_module_qn = self._file_to_module_qn(
            parsed_file.file_info.relative_path
        )
        self._current_language = parsed_file.file_info.language.value

        await self.client.execute(
            FileQueries.CREATE_FILE,
            {
                "path": file_path,
                "name": parsed_file.file_info.path.name,
                "language": parsed_file.file_info.language.value,
                "hash": parsed_file.file_info.content_hash,
                "line_count": parsed_file.file_info.line_count,
                "summary": parsed_file.summary,
            },
        )

        for imp in parsed_file.imports:
            await self.client.execute(
                EntityQueries.CREATE_IMPORT,
                {
                    "name": imp.name,
                    "file_path": file_path,
                    "alias": imp.alias,
                    "source": imp.source,
                    "is_external": imp.is_external,
                    "line_number": imp.line_number,
                },
            )
            await self.client.execute(
                RelationshipQueries.CREATE_FILE_IMPORTS,
                {"file_path": file_path, "import_name": imp.name},
            )

        for entity in parsed_file.entities:
            await self._create_entity(entity, file_path)

    async def _create_entity(
        self,
        entity: CodeEntity,
        file_path: str,
        parent_class: str | None = None,
    ) -> None:
        """Create a code entity node and its relationships.

        Args:
            entity: Code entity to create.
            file_path: File path containing the entity.
            parent_class: Parent class name for methods.
        """
        if entity.type == EntityType.CLASS:
            await self._create_class(entity, file_path)
        elif entity.type == EntityType.FUNCTION:
            await self._create_function(entity, file_path)
        elif entity.type == EntityType.METHOD:
            await self._create_method(entity, file_path, parent_class)

    def _build_base_entity_properties(
        self,
        entity: CodeEntity,
        file_path: str,
    ) -> dict:
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

    async def _create_defines_relationship(
        self,
        file_path: str,
        entity_type: str,
        entity_name: str,
    ) -> None:
        """Create a DEFINES relationship between file and entity.

        Args:
            file_path: File path.
            entity_type: Entity type (class, function).
            entity_name: Entity qualified name.
        """
        if entity_type == "class":
            query = RelationshipQueries.CREATE_FILE_DEFINES_CLASS
            params = {"file_path": file_path, "class_name": entity_name}
        elif entity_type == "function":
            query = RelationshipQueries.CREATE_FILE_DEFINES_FUNCTION
            params = {"file_path": file_path, "function_name": entity_name}
        else:
            logger.warning(f"Unknown entity type for DEFINES relationship: {entity_type}")
            return

        await self.client.execute(query, params)

    def _file_to_module_qn(self, relative_path: str) -> str:
        """Convert file path to module qualified name."""
        path = Path(relative_path)
        parts = list(path.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        base = f"{self.project_name}.{'.'.join(parts)}" if parts else self.project_name
        return base if self.project_name else ".".join(parts)

    async def _create_calls_relationships(
        self,
        caller_name: str,
        calls_list: list[str],
        class_context: str | None = None,
    ) -> None:
        """Create CALLS relationships for an entity.

        Uses CallProcessor for enhanced resolution when available.

        Args:
            caller_name: Caller qualified name.
            calls_list: List of callee names.
            class_context: Enclosing class for method context.
        """
        for call in calls_list:
            resolved_qn = None

            # Try enhanced resolution with CallProcessor
            if self.call_processor and self._current_module_qn:
                try:
                    result = self.call_processor.resolve_call(
                        call_name=call,
                        module_qn=self._current_module_qn,
                        class_context=class_context,
                        language=getattr(self, '_current_language', 'python'),
                    )
                    if result:
                        entity_type, resolved_qn = result
                        logger.debug(f"CallProcessor resolved: {call} -> {resolved_qn}")
                except Exception as e:
                    logger.debug(f"CallProcessor resolution failed for {call}: {e}")

            # Use resolved name or fall back to original
            callee_name = resolved_qn or call

            try:
                # Try exact match by qualified name
                await self.client.execute(
                    RelationshipQueries.CREATE_FUNCTION_CALLS,
                    {"caller_name": caller_name, "callee_name": callee_name},
                )
            except Exception as e:
                logger.debug(
                    f"Exact CALLS match failed from {caller_name} to {callee_name}: {e}"
                )

            # For method calls like "obj.method", also link by method name
            # This handles polymorphic calls where we don't know the exact class
            if "." in call:
                method_name = call.split(".")[-1]
                # Skip common non-method calls (module.func patterns)
                if method_name and not method_name.startswith("_"):
                    try:
                        await self.client.execute(
                            RelationshipQueries.CREATE_METHOD_CALLS_BY_NAME,
                            {"caller_name": caller_name, "method_name": method_name},
                        )
                    except Exception as e:
                        logger.debug(
                            f"Method name CALLS match failed from {caller_name} to {method_name}: {e}"
                        )

    async def _create_class(self, entity: CodeEntity, file_path: str) -> None:
        """Create a class node and its relationships.

        Args:
            entity: Class entity.
            file_path: File path.
        """
        properties = self._build_base_entity_properties(entity, file_path)
        await self.client.execute(EntityQueries.CREATE_CLASS, properties)

        await self._create_defines_relationship(
            file_path,
            "class",
            entity.qualified_name,
        )

        for base_class in entity.base_classes:
            try:
                await self.client.execute(
                    RelationshipQueries.CREATE_CLASS_EXTENDS,
                    {"child_name": entity.qualified_name, "parent_name": base_class},
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create EXTENDS relationship for {entity.qualified_name} -> {base_class}: {e}"
                )

        for child in entity.children:
            if child.type == EntityType.METHOD:
                await self._create_method(child, file_path, entity.qualified_name)

    async def _create_function(self, entity: CodeEntity, file_path: str) -> None:
        """Create a function node and its relationships.

        Args:
            entity: Function entity.
            file_path: File path.
        """
        properties = self._build_base_entity_properties(entity, file_path)
        properties["is_async"] = entity.is_async

        await self.client.execute(EntityQueries.CREATE_FUNCTION, properties)

        await self._create_defines_relationship(
            file_path,
            "function",
            entity.qualified_name,
        )

        # Functions don't have class context
        await self._create_calls_relationships(entity.qualified_name, entity.calls)

    async def _create_method(
        self,
        entity: CodeEntity,
        file_path: str,
        parent_class: str | None,
    ) -> None:
        """Create a method node and its relationships.

        Args:
            entity: Method entity.
            file_path: File path.
            parent_class: Parent class qualified name.
        """
        class_name = parent_class or entity.parent_class

        properties = self._build_base_entity_properties(entity, file_path)
        properties.update({
            "is_async": entity.is_async,
            "is_static": entity.is_static,
            "is_classmethod": entity.is_classmethod,
            "parent_class": class_name,
        })

        await self.client.execute(EntityQueries.CREATE_METHOD, properties)

        if class_name:
            try:
                await self.client.execute(
                    RelationshipQueries.CREATE_CLASS_DEFINES_METHOD,
                    {"class_name": class_name, "method_name": entity.qualified_name},
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create DEFINES_METHOD relationship for {class_name} -> {entity.qualified_name}: {e}"
                )

        # Pass class context for better super() and inherited method resolution
        await self._create_calls_relationships(
            entity.qualified_name, entity.calls, class_context=class_name
        )
