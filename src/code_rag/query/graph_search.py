"""Graph-based search using Memgraph."""

import logging
from dataclasses import dataclass

from code_rag.core.errors import GraphError, QueryError
from code_rag.core.types import EntityType
from code_rag.graph.client import MemgraphClient
from code_rag.graph.queries import SearchQueries
from code_rag.graph.statistics import GraphStatistics

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_LIMIT = 10
DEFAULT_MAX_DEPTH = 2
RELATED_ENTITIES_LIMIT = 20


@dataclass
class EntitySearchResult:
    """Result from entity search."""

    name: str
    qualified_name: str
    entity_type: str
    file_path: str
    summary: str | None = None
    signature: str | None = None
    start_line: int | None = None
    end_line: int | None = None


@dataclass
class RelatedEntityResult:
    """Result from related entity search."""

    name: str
    qualified_name: str
    entity_type: str
    file_path: str
    summary: str | None
    distance: int


class GraphSearcher:
    """Performs graph-based searches on the knowledge graph."""

    def __init__(self, client: MemgraphClient):
        """Initialize graph searcher.

        Args:
            client: Memgraph client instance.
        """
        self.client = client

    async def find_entity_by_name(
        self,
        name: str,
        entity_type: str | None = None,
    ) -> list[dict]:
        """Find entities by name.

        Args:
            name: Entity name to search for.
            entity_type: Optional filter by type (Class, Function, Method).

        Returns:
            List of matching entities.

        Raises:
            QueryError: If search fails or input is invalid.
        """
        if not name or not name.strip():
            raise QueryError("Entity name cannot be empty")

        if entity_type:
            self._validate_entity_type(entity_type)

        try:
            logger.debug(f"Searching for entity: {name}, type: {entity_type}")
            query = self._build_entity_query(entity_type)
            results = await self.client.execute(query, {"name": name})
            logger.debug(f"Found {len(results)} entities")
            return results

        except GraphError as e:
            logger.error(f"Graph error finding entity: {e}")
            raise QueryError(f"Failed to find entity: {name}", cause=e)

    async def find_callers(self, function_name: str) -> list[dict]:
        """Find all callers of a function.

        Args:
            function_name: Name of the function to find callers for.

        Returns:
            List of caller entities.

        Raises:
            QueryError: If search fails.
        """
        if not function_name or not function_name.strip():
            raise QueryError("Function name cannot be empty")

        try:
            logger.debug(f"Finding callers of: {function_name}")
            results = await self.client.execute(
                SearchQueries.FIND_CALLERS,
                {"qualified_name": function_name},
            )
            logger.debug(f"Found {len(results)} callers")
            return results

        except GraphError as e:
            logger.error(f"Graph error finding callers: {e}")
            raise QueryError(f"Failed to find callers for: {function_name}", cause=e)

    async def find_callees(self, function_name: str) -> list[dict]:
        """Find all functions called by a function.

        Args:
            function_name: Name of the calling function.

        Returns:
            List of called entities.

        Raises:
            QueryError: If search fails.
        """
        if not function_name or not function_name.strip():
            raise QueryError("Function name cannot be empty")

        try:
            logger.debug(f"Finding callees of: {function_name}")
            results = await self.client.execute(
                SearchQueries.FIND_CALLEES,
                {"qualified_name": function_name},
            )
            logger.debug(f"Found {len(results)} callees")
            return results

        except GraphError as e:
            logger.error(f"Graph error finding callees: {e}")
            raise QueryError(f"Failed to find callees for: {function_name}", cause=e)

    async def find_class_hierarchy(self, class_name: str) -> list[dict]:
        """Find the inheritance hierarchy of a class.

        Args:
            class_name: Name of the class.

        Returns:
            Class hierarchy information.

        Raises:
            QueryError: If search fails.
        """
        if not class_name or not class_name.strip():
            raise QueryError("Class name cannot be empty")

        try:
            logger.debug(f"Finding class hierarchy for: {class_name}")
            results = await self.client.execute(
                SearchQueries.FIND_CLASS_HIERARCHY,
                {"qualified_name": class_name},
            )
            logger.debug(f"Found {len(results)} hierarchy results")
            return results

        except GraphError as e:
            logger.error(f"Graph error finding class hierarchy: {e}")
            raise QueryError(f"Failed to find hierarchy for: {class_name}", cause=e)

    async def find_file_dependencies(self, file_path: str) -> list[dict]:
        """Find dependencies (imports) of a file.

        Args:
            file_path: Path to the file.

        Returns:
            List of imports.

        Raises:
            QueryError: If search fails.
        """
        if not file_path or not file_path.strip():
            raise QueryError("File path cannot be empty")

        try:
            logger.debug(f"Finding dependencies for file: {file_path}")
            results = await self.client.execute(
                SearchQueries.FIND_FILE_DEPENDENCIES,
                {"path": file_path},
            )
            logger.debug(f"Found {len(results)} dependencies")
            return results

        except GraphError as e:
            logger.error(f"Graph error finding file dependencies: {e}")
            raise QueryError(f"Failed to find dependencies for: {file_path}", cause=e)

    async def get_file_entities(self, file_path: str) -> list[dict]:
        """Get all entities defined in a file.

        Args:
            file_path: Path to the file.

        Returns:
            List of entities in the file.

        Raises:
            QueryError: If search fails.
        """
        if not file_path or not file_path.strip():
            raise QueryError("File path cannot be empty")

        try:
            logger.debug(f"Getting entities for file: {file_path}")
            results = await self.client.execute(
                SearchQueries.GET_FILE_ENTITIES,
                {"path": file_path},
            )
            logger.debug(f"Found {len(results)} entities")
            return results

        except GraphError as e:
            logger.error(f"Graph error getting file entities: {e}")
            raise QueryError(f"Failed to get entities for: {file_path}", cause=e)

    async def search_by_name(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[dict]:
        """Search entities by name substring.

        Args:
            query: Search query.
            limit: Maximum results.

        Returns:
            Matching entities.

        Raises:
            QueryError: If search fails.
        """
        if not query or not query.strip():
            raise QueryError("Search query cannot be empty")

        try:
            logger.debug(f"Searching by name: {query}, limit: {limit}")
            results = await self.client.execute(
                SearchQueries.SEARCH_BY_NAME,
                {"query": query, "limit": limit},
            )
            logger.debug(f"Found {len(results)} matches")
            return results

        except GraphError as e:
            logger.error(f"Graph error searching by name: {e}")
            raise QueryError(f"Failed to search for: {query}", cause=e)

    async def find_related_entities(
        self,
        entity_name: str,
        max_depth: int = DEFAULT_MAX_DEPTH,
    ) -> list[dict]:
        """Find entities related to a given entity.

        Args:
            entity_name: Name of the source entity.
            max_depth: Maximum relationship depth.

        Returns:
            Related entities.

        Raises:
            QueryError: If search fails.
        """
        if not entity_name or not entity_name.strip():
            raise QueryError("Entity name cannot be empty")

        if max_depth < 1:
            raise QueryError("Max depth must be at least 1")

        try:
            logger.debug(f"Finding related entities for: {entity_name}, depth: {max_depth}")
            query = f"""
            MATCH (source)
            WHERE source.name = $name OR source.qualified_name = $name
            MATCH path = (source)-[*1..{max_depth}]-(related)
            WHERE source <> related
            RETURN DISTINCT related.name as name,
                   related.qualified_name as qualified_name,
                   labels(related)[0] as type,
                   related.file_path as file_path,
                   related.summary as summary,
                   length(path) as distance
            ORDER BY distance
            LIMIT {RELATED_ENTITIES_LIMIT}
            """
            results = await self.client.execute(query, {"name": entity_name})
            logger.debug(f"Found {len(results)} related entities")
            return results

        except GraphError as e:
            logger.error(f"Graph error finding related entities: {e}")
            raise QueryError(f"Failed to find related entities for: {entity_name}", cause=e)

    async def get_statistics(self) -> dict:
        """Get graph statistics.

        Returns:
            Statistics dictionary.

        Raises:
            QueryError: If statistics retrieval fails.
        """
        try:
            logger.debug("Getting graph statistics")
            stats = GraphStatistics(self.client)
            return await stats.get_entity_counts()

        except GraphError as e:
            logger.error(f"Graph error getting statistics: {e}")
            raise QueryError("Failed to get graph statistics", cause=e)

    def _validate_entity_type(self, entity_type: str) -> None:
        """Validate entity type.

        Args:
            entity_type: Entity type to validate.

        Raises:
            QueryError: If entity type is invalid.
        """
        valid_types = {e.value for e in EntityType} | {"Class", "Function", "Method"}
        if entity_type not in valid_types:
            raise QueryError(f"Invalid entity type: {entity_type}")

    def _build_entity_query(self, entity_type: str | None) -> str:
        """Build Cypher query for entity search.

        Args:
            entity_type: Optional entity type filter.

        Returns:
            Cypher query string.
        """
        if entity_type:
            return f"""
            MATCH (n:{entity_type})
            WHERE n.name = $name OR n.qualified_name = $name
            RETURN n.name as name,
                   n.qualified_name as qualified_name,
                   n.file_path as file_path,
                   n.summary as summary,
                   n.signature as signature,
                   n.start_line as start_line,
                   n.end_line as end_line
            """
        else:
            return """
            MATCH (n)
            WHERE n.name = $name OR n.qualified_name = $name
            RETURN n.name as name,
                   n.qualified_name as qualified_name,
                   labels(n)[0] as type,
                   n.file_path as file_path,
                   n.summary as summary,
                   n.signature as signature,
                   n.start_line as start_line,
                   n.end_line as end_line
            """
