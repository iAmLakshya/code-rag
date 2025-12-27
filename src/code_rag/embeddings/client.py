import logging
from enum import Enum
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from code_rag.config import get_settings
from code_rag.core.errors import VectorStoreError

logger = logging.getLogger(__name__)


class CollectionName(str, Enum):
    CODE_CHUNKS = "code_chunks"
    SUMMARIES = "summaries"


class QdrantManager:
    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        grpc_port: int | None = None,
    ):
        settings = get_settings()
        self._host = host or settings.qdrant_host
        self._port = port or settings.qdrant_port
        self._grpc_port = grpc_port or settings.qdrant_grpc_port
        self._dimensions = settings.embedding_dimensions
        self._client: AsyncQdrantClient | None = None

    async def connect(self) -> None:
        if self._client is None:
            try:
                self._client = AsyncQdrantClient(
                    host=self._host,
                    port=self._port,
                    grpc_port=self._grpc_port,
                    prefer_grpc=True,
                )
                logger.info(
                    f"Connected to Qdrant at {self._host}:{self._port} (gRPC: {self._grpc_port})"
                )
            except Exception as e:
                raise VectorStoreError("Failed to connect to Qdrant", cause=e)

    async def close(self) -> None:
        if self._client:
            try:
                await self._client.close()
                logger.info("Closed Qdrant connection")
            except Exception as e:
                logger.warning(f"Error closing Qdrant connection: {e}")
            finally:
                self._client = None

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise VectorStoreError("Client not connected. Call connect() first.")
        return self._client

    async def health_check(self) -> bool:
        try:
            await self.client.get_collections()
            logger.debug("Qdrant health check passed")
            return True
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")
            return False

    async def create_collections(self) -> None:
        try:
            collections = await self.client.get_collections()
            existing = {c.name for c in collections.collections}

            if CollectionName.CODE_CHUNKS.value not in existing:
                await self._create_collection_with_indexes(
                    CollectionName.CODE_CHUNKS.value,
                    ["file_path", "entity_type", "language", "content_hash", "project_name"],
                )
                logger.info(f"Created collection: {CollectionName.CODE_CHUNKS.value}")

            if CollectionName.SUMMARIES.value not in existing:
                await self._create_collection_with_indexes(
                    CollectionName.SUMMARIES.value,
                    ["file_path", "entity_type"],
                )
                logger.info(f"Created collection: {CollectionName.SUMMARIES.value}")
        except Exception as e:
            raise VectorStoreError("Failed to create collections", cause=e)

    async def _create_collection_with_indexes(
        self, name: str, index_fields: list[str]
    ) -> None:
        await self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=self._dimensions,
                distance=models.Distance.COSINE,
            ),
        )
        await self._create_keyword_indexes(name, index_fields)

    async def _create_keyword_indexes(
        self, collection: str, fields: list[str]
    ) -> None:
        for field in fields:
            await self.client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    async def upsert(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        try:
            points = [
                models.PointStruct(id=id_, vector=vector, payload=payload)
                for id_, vector, payload in zip(ids, vectors, payloads)
            ]
            await self.client.upsert(collection_name=collection, points=points)
            logger.debug(f"Upserted {len(points)} vectors to {collection}")
        except Exception as e:
            raise VectorStoreError(f"Failed to upsert vectors to {collection}", cause=e)

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        try:
            query_filter = self._build_filter(filters) if filters else None

            response = await self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )

            results = [
                {"id": str(point.id), "score": point.score, "payload": point.payload}
                for point in response.points
            ]
            logger.debug(f"Found {len(results)} results in {collection}")
            return results
        except Exception as e:
            raise VectorStoreError(f"Failed to search {collection}", cause=e)

    async def delete(self, collection: str, filters: dict[str, Any]) -> None:
        try:
            await self.client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(
                    filter=self._build_filter(filters)
                ),
            )
            logger.debug(f"Deleted vectors from {collection} with filters: {filters}")
        except Exception as e:
            raise VectorStoreError(f"Failed to delete from {collection}", cause=e)

    def _build_filter(self, conditions: dict[str, Any]) -> models.Filter:
        must_conditions = [
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in conditions.items()
        ]
        return models.Filter(must=must_conditions)

    async def file_needs_update(
        self, collection: str, file_path: str, content_hash: str
    ) -> bool:
        try:
            result = await self.client.scroll(
                collection_name=collection,
                scroll_filter=self._build_filter({"file_path": file_path}),
                limit=1,
                with_payload=["content_hash"],
            )

            points = result[0]
            if not points:
                logger.debug(f"File {file_path} not found in {collection}")
                return True

            existing_hash = points[0].payload.get("content_hash")
            needs_update = existing_hash != content_hash
            logger.debug(
                f"File {file_path} {'needs' if needs_update else 'does not need'} update"
            )
            return needs_update
        except Exception as e:
            logger.warning(f"Error checking file update status: {e}")
            return True

    async def get_collection_info(self, collection: str) -> models.CollectionInfo:
        try:
            return await self.client.get_collection(collection)
        except Exception as e:
            raise VectorStoreError(
                f"Failed to get collection info for {collection}", cause=e
            )

    async def clear_collections(self) -> None:
        for collection in [CollectionName.CODE_CHUNKS, CollectionName.SUMMARIES]:
            try:
                await self.client.delete_collection(collection.value)
                logger.info(f"Deleted collection: {collection.value}")
            except Exception as e:
                logger.debug(
                    f"Collection {collection.value} does not exist or already deleted: {e}"
                )
        await self.create_collections()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
