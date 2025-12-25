"""Graph schema definitions and setup."""

import logging

from code_rag.core.errors import GraphError
from code_rag.graph.client import MemgraphClient

logger = logging.getLogger(__name__)


class GraphSchema:
    """Manages the graph database schema."""

    INDEX_DEFINITIONS = [
        ("Project", "name"),
        ("File", "path"),
        ("File", "hash"),
        ("Class", "name"),
        ("Class", "qualified_name"),
        ("Function", "name"),
        ("Function", "qualified_name"),
        ("Method", "name"),
        ("Method", "qualified_name"),
        ("Import", "name"),
    ]

    def __init__(self, client: MemgraphClient):
        """Initialize schema manager.

        Args:
            client: Memgraph client instance.
        """
        self.client = client

    def _generate_index_queries(self) -> list[str]:
        """Generate index creation queries from definitions.

        Returns:
            List of CREATE INDEX queries.
        """
        return [
            f"CREATE INDEX ON :{label}({property});"
            for label, property in self.INDEX_DEFINITIONS
        ]

    async def setup(self) -> None:
        """Create all indexes and constraints."""
        queries = self._generate_index_queries()
        for query in queries:
            try:
                await self.client.execute(query)
            except Exception as e:
                logger.warning(f"Index creation skipped (may already exist): {e}")

    async def reset(self) -> None:
        """Clear all data and recreate schema."""
        await self.client.clear_database()
        await self.setup()

    async def get_schema_info(self) -> dict:
        """Get information about the current schema.

        Returns:
            Dictionary with node and relationship counts.
        """
        node_counts = await self._get_node_counts()
        rel_counts = await self._get_relationship_counts()

        return {
            "nodes": node_counts,
            "relationships": rel_counts,
            "total_nodes": sum(node_counts.values()),
            "total_relationships": sum(rel_counts.values()),
        }

    async def _get_node_counts(self) -> dict[str, int]:
        """Get counts of nodes by label using single aggregation query.

        Returns:
            Dictionary mapping label to count.
        """
        query = """
        CALL db.labels() YIELD label
        WITH label
        CALL {
            WITH label
            WITH label, '(n:' + label + ')' AS pattern
            WITH label, 'MATCH ' + pattern + ' RETURN count(n) AS count' AS query
            RETURN label, query
        }
        RETURN label, query
        """

        try:
            labels_result = await self.client.execute(query)
            counts = {}

            for record in labels_result:
                label = record["label"]
                count_query = record["query"]
                try:
                    result = await self.client.execute(count_query)
                    counts[label] = result[0]["count"] if result else 0
                except Exception as e:
                    logger.warning(f"Failed to get count for label {label}: {e}")
                    counts[label] = 0

            return counts
        except Exception as e:
            logger.warning(f"Failed to retrieve node counts: {e}")
            return {}

    async def _get_relationship_counts(self) -> dict[str, int]:
        """Get counts of relationships by type using single aggregation query.

        Returns:
            Dictionary mapping relationship type to count.
        """
        query = """
        CALL db.relationshipTypes() YIELD relationshipType
        WITH relationshipType
        CALL {
            WITH relationshipType
            WITH relationshipType, '()-[r:' + relationshipType + ']-()' AS pattern
            WITH relationshipType, 'MATCH ' + pattern + ' RETURN count(r) AS count' AS query
            RETURN relationshipType, query
        }
        RETURN relationshipType, query
        """

        try:
            types_result = await self.client.execute(query)
            counts = {}

            for record in types_result:
                rel_type = record["relationshipType"]
                count_query = record["query"]
                try:
                    result = await self.client.execute(count_query)
                    counts[rel_type] = result[0]["count"] if result else 0
                except Exception as e:
                    logger.warning(f"Failed to get count for relationship type {rel_type}: {e}")
                    counts[rel_type] = 0

            return counts
        except Exception as e:
            logger.warning(f"Failed to retrieve relationship counts: {e}")
            return {}
