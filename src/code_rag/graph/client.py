"""Memgraph client for graph database operations."""

import logging
from contextlib import asynccontextmanager
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession

from code_rag.config import get_settings
from code_rag.core.errors import ConnectionError, GraphError

logger = logging.getLogger(__name__)


class MemgraphClient:
    """Async client for Memgraph graph database."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        """Initialize Memgraph client.

        Args:
            uri: Memgraph connection URI. Defaults to settings.
            user: Username. Defaults to settings.
            password: Password. Defaults to settings.
        """
        settings = get_settings()
        self._uri = uri or settings.memgraph_uri
        self._user = user or settings.memgraph_user
        self._password = password or settings.memgraph_password
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Establish connection to Memgraph."""
        if self._driver is None:
            try:
                self._driver = AsyncGraphDatabase.driver(
                    self._uri,
                    auth=(self._user, self._password),
                )
                await self._driver.verify_connectivity()
                logger.info(f"Connected to Memgraph at {self._uri}")
            except Exception as e:
                raise ConnectionError(
                    f"Failed to connect to Memgraph at {self._uri}",
                    cause=e,
                ) from e

    async def close(self) -> None:
        """Close the database connection."""
        if self._driver:
            try:
                await self._driver.close()
                self._driver = None
                logger.info("Closed Memgraph connection")
            except Exception as e:
                logger.warning(f"Error closing Memgraph connection: {e}")

    @asynccontextmanager
    async def session(self):
        """Get a database session context manager."""
        if self._driver is None:
            await self.connect()
        session: AsyncSession = self._driver.session()
        try:
            yield session
        finally:
            await session.close()

    def _normalize_parameters(self, parameters: dict[str, Any] | None) -> dict[str, Any]:
        """Normalize parameters to avoid None values."""
        return parameters or {}

    async def execute(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results.

        Args:
            query: Cypher query string.
            parameters: Query parameters.

        Returns:
            List of result records as dictionaries.

        Raises:
            GraphError: If query execution fails.
        """
        try:
            async with self.session() as session:
                result = await session.run(query, self._normalize_parameters(parameters))
                records = await result.data()
                return records
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise GraphError(f"Failed to execute query: {query[:100]}...", cause=e) from e

    async def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a write query in a transaction.

        Args:
            query: Cypher query string.
            parameters: Query parameters.

        Returns:
            List of result records as dictionaries.

        Raises:
            GraphError: If query execution fails.
        """
        try:
            async with self.session() as session:
                result = await session.execute_write(
                    lambda tx: tx.run(query, self._normalize_parameters(parameters)).data()
                )
                return result
        except Exception as e:
            logger.error(f"Write query execution failed: {e}")
            raise GraphError(f"Failed to execute write query: {query[:100]}...", cause=e) from e

    async def health_check(self) -> bool:
        """Check if the database is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            await self.execute("RETURN 1 as n")
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def clear_database(self) -> None:
        """Clear all nodes and relationships from the database.

        Raises:
            GraphError: If clearing fails.
        """
        try:
            await self.execute("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared successfully")
        except Exception as e:
            raise GraphError("Failed to clear database", cause=e) from e

    async def get_node_count(self) -> int:
        """Get total node count."""
        result = await self.execute("MATCH (n) RETURN count(n) as count")
        return result[0]["count"] if result else 0

    async def get_relationship_count(self) -> int:
        """Get total relationship count."""
        result = await self.execute("MATCH ()-[r]->() RETURN count(r) as count")
        return result[0]["count"] if result else 0

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
