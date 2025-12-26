"""Tests for database connections."""

import pytest

from code_rag.graph.client import MemgraphClient
from code_rag.embeddings.client import QdrantManager, CollectionName


class TestMemgraphConnection:
    """Tests for Memgraph database connection."""

    @pytest.mark.asyncio
    async def test_connect_to_memgraph(self):
        """Test connecting to Memgraph."""
        client = MemgraphClient()
        try:
            await client.connect()
            assert await client.health_check(), "Memgraph should be healthy"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_execute_query(self):
        """Test executing a simple query."""
        async with MemgraphClient() as client:
            result = await client.execute("RETURN 1 + 1 AS result")
            assert result[0]["result"] == 2

    @pytest.mark.asyncio
    async def test_create_and_query_node(self):
        """Test creating and querying a node."""
        async with MemgraphClient() as client:
            # Create a test node
            await client.execute(
                "CREATE (t:TestNode {name: $name})",
                {"name": "test_node_123"}
            )

            # Query the node
            result = await client.execute(
                "MATCH (t:TestNode {name: $name}) RETURN t.name as name",
                {"name": "test_node_123"}
            )
            assert len(result) == 1
            assert result[0]["name"] == "test_node_123"

            # Clean up
            await client.execute(
                "MATCH (t:TestNode {name: $name}) DELETE t",
                {"name": "test_node_123"}
            )

    @pytest.mark.asyncio
    async def test_get_node_count(self):
        """Test getting node count."""
        async with MemgraphClient() as client:
            count = await client.get_node_count()
            assert isinstance(count, int)
            assert count >= 0


class TestQdrantConnection:
    """Tests for Qdrant database connection."""

    @pytest.mark.asyncio
    async def test_connect_to_qdrant(self):
        """Test connecting to Qdrant."""
        manager = QdrantManager()
        try:
            await manager.connect()
            assert await manager.health_check(), "Qdrant should be healthy"
        finally:
            await manager.close()

    @pytest.mark.asyncio
    async def test_create_collections(self):
        """Test creating collections."""
        async with QdrantManager() as manager:
            await manager.create_collections()

            collections = await manager.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            assert CollectionName.CODE_CHUNKS.value in collection_names
            assert CollectionName.SUMMARIES.value in collection_names

    @pytest.mark.asyncio
    async def test_upsert_and_search_vectors(self):
        """Test upserting and searching vectors."""
        import uuid

        async with QdrantManager() as manager:
            await manager.create_collections()

            test_vector = [0.1] * 1536
            test_id = str(uuid.uuid4())
            test_payload = {
                "file_path": "/test/file.py",
                "entity_type": "function",
                "entity_name": "test_func",
                "language": "python",
                "content": "def test(): pass",
            }

            await manager.upsert(
                collection=CollectionName.CODE_CHUNKS.value,
                ids=[test_id],
                vectors=[test_vector],
                payloads=[test_payload],
            )

            results = await manager.search(
                collection=CollectionName.CODE_CHUNKS.value,
                query_vector=test_vector,
                limit=1,
            )

            assert len(results) >= 1
            assert results[0]["payload"]["entity_name"] == "test_func"

            await manager.delete(
                collection=CollectionName.CODE_CHUNKS.value,
                filters={"file_path": "/test/file.py"},
            )


class TestDatabaseIntegration:
    """Integration tests for databases."""

    @pytest.mark.asyncio
    async def test_both_databases_accessible(self):
        """Test that both databases are accessible."""
        # Test Memgraph
        async with MemgraphClient() as memgraph:
            mg_healthy = await memgraph.health_check()

        # Test Qdrant
        async with QdrantManager() as qdrant:
            qd_healthy = await qdrant.health_check()

        assert mg_healthy, "Memgraph should be accessible"
        assert qd_healthy, "Qdrant should be accessible"
