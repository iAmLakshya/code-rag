"""Data access layer with Repository pattern."""

from code_rag.tests.fixtures.sample_project.src.repositories.base import (
    Repository,
    InMemoryRepository,
    CachedRepository,
    UnitOfWork,
    Specification,
    AndSpecification,
    OrSpecification,
    NotSpecification,
    QueryResult,
)

__all__ = [
    "Repository",
    "InMemoryRepository",
    "CachedRepository",
    "UnitOfWork",
    "Specification",
    "AndSpecification",
    "OrSpecification",
    "NotSpecification",
    "QueryResult",
]
