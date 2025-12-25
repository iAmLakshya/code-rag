"""Project manager for Code RAG."""

import logging
from typing import Any

from code_rag.embeddings.client import QdrantManager
from code_rag.graph.client import MemgraphClient
from code_rag.projects.cleanup import ProjectCleanupService
from code_rag.projects.models import Project
from code_rag.projects.repository import ProjectRepository

logger = logging.getLogger(__name__)


class ProjectManager:
    """Manages Code RAG projects and their indexes."""

    def __init__(
        self,
        memgraph: MemgraphClient | None = None,
        qdrant: QdrantManager | None = None,
    ):
        self._memgraph = memgraph
        self._qdrant = qdrant
        self._repository: ProjectRepository | None = None
        self._cleanup_service: ProjectCleanupService | None = None
        self._owns_connections = memgraph is None and qdrant is None

    async def connect(self) -> None:
        if self._owns_connections:
            self._memgraph = MemgraphClient()
            await self._memgraph.connect()

            self._qdrant = QdrantManager()
            await self._qdrant.connect()

        if not self._memgraph or not self._qdrant:
            raise RuntimeError("Database clients not initialized")

        self._repository = ProjectRepository(self._memgraph)
        self._cleanup_service = ProjectCleanupService(self._qdrant)

        logger.info("Project manager connected to databases")

    async def close(self) -> None:
        if self._owns_connections:
            if self._memgraph:
                await self._memgraph.close()
            if self._qdrant:
                await self._qdrant.close()

        logger.info("Project manager disconnected from databases")

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def list_projects(self) -> list[Project]:
        self._ensure_connected()
        return await self._repository.list_all()

    async def get_project(self, name: str) -> Project | None:
        self._ensure_connected()
        return await self._repository.get_by_name(name)

    async def delete_project(self, name: str) -> bool:
        self._ensure_connected()

        project = await self._repository.get_by_name(name)
        if not project:
            logger.warning(f"Project '{name}' not found for deletion")
            return False

        for index in project.indexes:
            await self._cleanup_service.delete_from_qdrant(index.path)

        await self._repository.delete(name)

        logger.info(f"Successfully deleted project '{name}'")
        return True

    async def delete_index(self, project_name: str, path: str) -> bool:
        self._ensure_connected()

        await self._cleanup_service.delete_from_qdrant(path)
        await self._repository.delete_index(path)
        await self._repository.delete_empty_project(project_name)

        logger.info(f"Successfully deleted index at path '{path}' from project '{project_name}'")
        return True

    async def get_project_stats(self, name: str) -> dict[str, Any]:
        self._ensure_connected()

        project = await self._repository.get_by_name(name)
        if not project:
            logger.warning(f"Project '{name}' not found for stats")
            return {}

        stats = {
            "name": project.name,
            "created_at": project.created_at,
            "last_indexed_at": project.last_indexed_at,
            "total_files": project.total_files,
            "total_entities": project.total_entities,
            "total_chunks": project.total_chunks,
            "indexes": [idx.to_dict() for idx in project.indexes],
        }

        logger.debug(f"Generated stats for project '{name}'")
        return stats

    async def get_chunk_count(self, path: str) -> int:
        self._ensure_connected()
        return await self._cleanup_service.get_chunk_count(path)

    def _ensure_connected(self) -> None:
        if not self._repository or not self._cleanup_service:
            raise RuntimeError("ProjectManager not connected. Call connect() first.")
