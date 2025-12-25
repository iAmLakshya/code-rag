"""Project management for Code RAG."""

from code_rag.projects.cleanup import ProjectCleanupService
from code_rag.projects.manager import ProjectManager
from code_rag.projects.models import Project, ProjectIndex
from code_rag.projects.repository import ProjectRepository

__all__ = [
    "ProjectManager",
    "Project",
    "ProjectIndex",
    "ProjectRepository",
    "ProjectCleanupService",
]
