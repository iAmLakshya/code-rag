"""Graph statistics and analytics."""

from code_rag.graph.client import MemgraphClient
from code_rag.graph.queries import SearchQueries


class GraphStatistics:
    """Provides statistical analysis of the knowledge graph."""

    def __init__(self, client: MemgraphClient):
        """Initialize statistics provider.

        Args:
            client: Memgraph client instance.
        """
        self.client = client

    async def get_entity_counts(self) -> dict:
        """Get counts of various entity types in the graph.

        Returns:
            Dictionary with counts of files, classes, functions, and methods.
        """
        result = await self.client.execute(SearchQueries.GET_STATS)
        if result:
            return result[0]
        return {
            "file_count": 0,
            "class_count": 0,
            "function_count": 0,
            "method_count": 0,
        }

    async def get_project_statistics(self, project_name: str) -> dict:
        """Get statistics for a specific project.

        Args:
            project_name: Name of the project.

        Returns:
            Dictionary with project-specific statistics.
        """
        query = """
        MATCH (p:Project {name: $project_name})
        OPTIONAL MATCH (f:File)
        WHERE f.path STARTS WITH p.path
        WITH p, count(DISTINCT f) as file_count
        OPTIONAL MATCH (c:Class)
        WHERE c.file_path STARTS WITH p.path
        WITH p, file_count, count(DISTINCT c) as class_count
        OPTIONAL MATCH (fn:Function)
        WHERE fn.file_path STARTS WITH p.path
        WITH p, file_count, class_count, count(DISTINCT fn) as function_count
        OPTIONAL MATCH (m:Method)
        WHERE m.file_path STARTS WITH p.path
        RETURN file_count, class_count, function_count, count(DISTINCT m) as method_count
        """

        result = await self.client.execute(query, {"project_name": project_name})
        if result:
            return result[0]
        return {
            "file_count": 0,
            "class_count": 0,
            "function_count": 0,
            "method_count": 0,
        }
