"""Service for cleaning up project data from databases."""

import logging

from qdrant_client import models

from code_rag.core.errors import VectorStoreError
from code_rag.embeddings.client import CollectionName, QdrantManager

logger = logging.getLogger(__name__)


class ProjectCleanupService:
    """Handles deletion of project data from databases."""

    def __init__(self, qdrant: QdrantManager):
        self._qdrant = qdrant

    async def delete_from_qdrant(self, path: str) -> int:
        total_deleted = 0

        for collection in [CollectionName.CODE_CHUNKS, CollectionName.SUMMARIES]:
            try:
                deleted = await self._delete_from_collection(collection, path)
                total_deleted += deleted
                logger.info(f"Deleted {deleted} points from {collection.value} for path '{path}'")
            except Exception as e:
                logger.error(
                    f"Failed to delete from collection {collection.value} for path '{path}'",
                    exc_info=True,
                )
                raise VectorStoreError(
                    f"Failed to delete from collection {collection.value}", cause=e
                ) from e

        return total_deleted

    async def get_chunk_count(self, path: str) -> int:
        try:
            filter_condition = self._build_path_filter(path)
            result = await self._qdrant.client.count(
                collection_name=CollectionName.CODE_CHUNKS.value,
                count_filter=filter_condition,
            )
            return result.count
        except Exception:
            logger.warning(f"Failed to get chunk count for path '{path}'", exc_info=True)
            return 0

    async def _delete_from_collection(self, collection: CollectionName, path: str) -> int:
        filter_condition = self._build_path_filter(path)

        await self._qdrant.client.delete(
            collection_name=collection.value,
            points_selector=models.FilterSelector(filter=filter_condition),
        )

        count_result = await self._qdrant.client.count(
            collection_name=collection.value,
            count_filter=filter_condition,
        )

        return count_result.count

    def _build_path_filter(self, path: str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="file_path",
                    match=models.MatchText(text=path),
                )
            ]
        )
