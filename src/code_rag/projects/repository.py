"""Repository for project data access."""

import logging
from datetime import datetime

from code_rag.core.errors import GraphError
from code_rag.graph.client import MemgraphClient
from code_rag.projects.models import Project, ProjectIndex

logger = logging.getLogger(__name__)


class ProjectRepository:
    """Data access layer for projects."""

    def __init__(self, memgraph: MemgraphClient):
        self._memgraph = memgraph

    async def list_all(self) -> list[Project]:
        query = """
        MATCH (p:Project)
        OPTIONAL MATCH (p)-[:CONTAINS]->(f:File)
        WITH p, count(DISTINCT f) as file_count
        RETURN p.name as name,
               p.path as path,
               p.created_at as created_at,
               p.last_indexed_at as last_indexed_at,
               file_count
        ORDER BY p.name
        """

        try:
            results = await self._memgraph.execute(query)
        except Exception as e:
            logger.error("Failed to list projects from Memgraph", exc_info=True)
            raise GraphError("Failed to list projects", cause=e) from e

        projects_dict: dict[str, dict] = {}

        for row in results:
            name = row.get("name", "unknown")
            path = row.get("path", "")
            file_count = row.get("file_count", 0)
            created_at = row.get("created_at")
            last_indexed_at = row.get("last_indexed_at")

            if name not in projects_dict:
                projects_dict[name] = {
                    "name": name,
                    "created_at": self._parse_datetime(created_at),
                    "last_indexed_at": self._parse_datetime(last_indexed_at),
                    "indexes": [],
                }

            if path:
                entity_count = await self._get_entity_count(path)
                projects_dict[name]["indexes"].append({
                    "path": path,
                    "file_count": file_count,
                    "entity_count": entity_count,
                    "indexed_at": self._parse_datetime(last_indexed_at),
                })

        return [
            Project(
                name=data["name"],
                created_at=data["created_at"],
                last_indexed_at=data["last_indexed_at"],
                indexes=tuple(ProjectIndex(**idx) for idx in data["indexes"]),
            )
            for data in projects_dict.values()
        ]

    async def get_by_name(self, name: str) -> Project | None:
        projects = await self.list_all()
        for project in projects:
            if project.name == name:
                return project
        return None

    async def delete(self, name: str) -> bool:
        project = await self.get_by_name(name)
        if not project:
            return False

        delete_query = """
        MATCH (p:Project {name: $name})
        OPTIONAL MATCH (f:File)
        WHERE f.path STARTS WITH p.path
        OPTIONAL MATCH (f)-[r1]->(e)
        OPTIONAL MATCH (e)-[r2]-()
        OPTIONAL MATCH (i:Import)
        WHERE i.file_path STARTS WITH p.path
        DELETE r1, r2, e, f, i, p
        """

        try:
            await self._memgraph.execute(delete_query, {"name": name})
            logger.info(f"Deleted project '{name}' from Memgraph")
            return True
        except Exception as e:
            logger.error(f"Failed to delete project '{name}' from Memgraph", exc_info=True)
            raise GraphError(f"Failed to delete project '{name}'", cause=e) from e

    async def delete_index(self, path: str) -> None:
        delete_query = """
        MATCH (f:File)
        WHERE f.path STARTS WITH $path
        OPTIONAL MATCH (f)-[r1]->(e)
        OPTIONAL MATCH (e)-[r2]-()
        DELETE r1, r2, e, f
        """

        try:
            await self._memgraph.execute(delete_query, {"path": path})
            logger.info(f"Deleted index at path '{path}' from Memgraph")
        except Exception as e:
            logger.error(f"Failed to delete index at path '{path}' from Memgraph", exc_info=True)
            raise GraphError(f"Failed to delete index at path '{path}'", cause=e) from e

    async def delete_empty_project(self, name: str) -> None:
        check_query = """
        MATCH (p:Project {name: $name})
        OPTIONAL MATCH (f:File)
        WHERE f.path STARTS WITH p.path
        WITH p, count(f) as remaining
        WHERE remaining = 0
        DELETE p
        """

        try:
            await self._memgraph.execute(check_query, {"name": name})
            logger.info(f"Deleted empty project '{name}' from Memgraph")
        except Exception as e:
            logger.error(f"Failed to check/delete empty project '{name}'", exc_info=True)
            raise GraphError(f"Failed to check/delete empty project '{name}'", cause=e) from e

    async def _get_entity_count(self, path: str) -> int:
        entity_query = """
        MATCH (n)
        WHERE (n:Class OR n:Function OR n:Method)
        AND n.file_path STARTS WITH $path
        RETURN count(n) as count
        """

        try:
            entity_result = await self._memgraph.execute(entity_query, {"path": path})
            return entity_result[0].get("count", 0) if entity_result else 0
        except Exception:
            logger.warning(f"Failed to get entity count for path '{path}'", exc_info=True)
            return 0

    def _parse_datetime(self, value) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                logger.warning(f"Failed to parse datetime: {value}")
                return None
        return None
