"""Repository pattern implementation with Unit of Work.

This module provides:
- Generic repository interface
- Unit of Work for transaction management
- Query builder for complex queries
- Specification pattern for filters
- Connection pooling management

Architecture:
    Service Layer
         ↓
    Repository Layer (this module)
         ↓
    Database Layer (SQLAlchemy/raw SQL)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    Callable,
    Generic,
    TypeVar,
    Protocol,
    Iterator,
    Sequence,
    runtime_checkable,
)
from contextlib import asynccontextmanager
import asyncio

from code_rag.tests.fixtures.sample_project.src.core.events import (
    EventBus,
    Event,
    EventType,
)
from code_rag.tests.fixtures.sample_project.src.core.cache import cached, get_cache


T = TypeVar("T")
ID = TypeVar("ID", str, int)


@runtime_checkable
class Entity(Protocol):
    """Protocol for entities that can be stored in repositories."""

    id: Any


@dataclass
class QueryResult(Generic[T]):
    """Result of a query with pagination metadata."""

    items: list[T]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool

    @property
    def pages(self) -> int:
        """Total number of pages."""
        return (self.total + self.page_size - 1) // self.page_size


class Specification(ABC, Generic[T]):
    """Specification pattern for composable query filters.

    Specifications can be combined using & (and), | (or), ~ (not).

    Example:
        active_users = ActiveSpec() & AgeSpec(min_age=18)
        users = await repo.find_by_spec(active_users)
    """

    @abstractmethod
    def is_satisfied_by(self, entity: T) -> bool:
        """Check if entity satisfies this specification."""
        pass

    @abstractmethod
    def to_query(self) -> dict[str, Any]:
        """Convert to database query format."""
        pass

    def __and__(self, other: "Specification[T]") -> "AndSpecification[T]":
        return AndSpecification(self, other)

    def __or__(self, other: "Specification[T]") -> "OrSpecification[T]":
        return OrSpecification(self, other)

    def __invert__(self) -> "NotSpecification[T]":
        return NotSpecification(self)


class AndSpecification(Specification[T]):
    """Combines two specifications with AND logic."""

    def __init__(self, left: Specification[T], right: Specification[T]):
        self.left = left
        self.right = right

    def is_satisfied_by(self, entity: T) -> bool:
        return (
            self.left.is_satisfied_by(entity)
            and self.right.is_satisfied_by(entity)
        )

    def to_query(self) -> dict[str, Any]:
        return {"$and": [self.left.to_query(), self.right.to_query()]}


class OrSpecification(Specification[T]):
    """Combines two specifications with OR logic."""

    def __init__(self, left: Specification[T], right: Specification[T]):
        self.left = left
        self.right = right

    def is_satisfied_by(self, entity: T) -> bool:
        return (
            self.left.is_satisfied_by(entity)
            or self.right.is_satisfied_by(entity)
        )

    def to_query(self) -> dict[str, Any]:
        return {"$or": [self.left.to_query(), self.right.to_query()]}


class NotSpecification(Specification[T]):
    """Negates a specification."""

    def __init__(self, spec: Specification[T]):
        self.spec = spec

    def is_satisfied_by(self, entity: T) -> bool:
        return not self.spec.is_satisfied_by(entity)

    def to_query(self) -> dict[str, Any]:
        return {"$not": self.spec.to_query()}


class Repository(ABC, Generic[T, ID]):
    """Abstract repository interface for CRUD operations.

    Repositories encapsulate data access logic and provide
    a collection-like interface for entities.
    """

    @abstractmethod
    async def get(self, id: ID) -> T | None:
        """Get entity by ID."""
        pass

    @abstractmethod
    async def get_many(self, ids: list[ID]) -> list[T]:
        """Get multiple entities by IDs."""
        pass

    @abstractmethod
    async def find(
        self,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        order_desc: bool = False,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[T]:
        """Find entities matching filters."""
        pass

    @abstractmethod
    async def find_one(self, filters: dict[str, Any]) -> T | None:
        """Find first entity matching filters."""
        pass

    @abstractmethod
    async def find_by_spec(self, spec: Specification[T]) -> list[T]:
        """Find entities matching a specification."""
        pass

    @abstractmethod
    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count entities matching filters."""
        pass

    @abstractmethod
    async def exists(self, id: ID) -> bool:
        """Check if entity exists."""
        pass

    @abstractmethod
    async def add(self, entity: T) -> T:
        """Add a new entity."""
        pass

    @abstractmethod
    async def add_many(self, entities: list[T]) -> list[T]:
        """Add multiple entities."""
        pass

    @abstractmethod
    async def update(self, entity: T) -> T:
        """Update an existing entity."""
        pass

    @abstractmethod
    async def delete(self, id: ID) -> bool:
        """Delete an entity by ID."""
        pass

    @abstractmethod
    async def delete_many(self, ids: list[ID]) -> int:
        """Delete multiple entities. Returns count deleted."""
        pass

    async def paginate(
        self,
        page: int = 1,
        page_size: int = 10,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        order_desc: bool = False,
    ) -> QueryResult[T]:
        """Get paginated results."""
        offset = (page - 1) * page_size

        items = await self.find(
            filters=filters,
            order_by=order_by,
            order_desc=order_desc,
            limit=page_size,
            offset=offset,
        )
        total = await self.count(filters)

        return QueryResult(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            has_next=offset + len(items) < total,
            has_prev=page > 1,
        )


class InMemoryRepository(Repository[T, ID]):
    """In-memory repository for testing and prototyping."""

    def __init__(self):
        self._storage: dict[ID, T] = {}
        self._id_counter = 0

    async def get(self, id: ID) -> T | None:
        return self._storage.get(id)

    async def get_many(self, ids: list[ID]) -> list[T]:
        return [self._storage[id] for id in ids if id in self._storage]

    async def find(
        self,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        order_desc: bool = False,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[T]:
        items = list(self._storage.values())

        # Apply filters
        if filters:
            items = [
                item for item in items
                if all(
                    getattr(item, k, None) == v
                    for k, v in filters.items()
                )
            ]

        # Apply sorting
        if order_by:
            items.sort(
                key=lambda x: getattr(x, order_by, None),
                reverse=order_desc,
            )

        # Apply pagination
        if offset:
            items = items[offset:]
        if limit:
            items = items[:limit]

        return items

    async def find_one(self, filters: dict[str, Any]) -> T | None:
        results = await self.find(filters, limit=1)
        return results[0] if results else None

    async def find_by_spec(self, spec: Specification[T]) -> list[T]:
        return [
            item for item in self._storage.values()
            if spec.is_satisfied_by(item)
        ]

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        if not filters:
            return len(self._storage)
        items = await self.find(filters)
        return len(items)

    async def exists(self, id: ID) -> bool:
        return id in self._storage

    async def add(self, entity: T) -> T:
        entity_id = getattr(entity, "id", None)
        if entity_id is None:
            self._id_counter += 1
            entity_id = self._id_counter
            setattr(entity, "id", entity_id)
        self._storage[entity_id] = entity
        return entity

    async def add_many(self, entities: list[T]) -> list[T]:
        return [await self.add(e) for e in entities]

    async def update(self, entity: T) -> T:
        entity_id = getattr(entity, "id")
        self._storage[entity_id] = entity
        return entity

    async def delete(self, id: ID) -> bool:
        if id in self._storage:
            del self._storage[id]
            return True
        return False

    async def delete_many(self, ids: list[ID]) -> int:
        count = 0
        for id in ids:
            if await self.delete(id):
                count += 1
        return count


class CachedRepository(Repository[T, ID]):
    """Repository decorator that adds caching.

    Uses cache for reads and invalidates on writes.
    """

    def __init__(
        self,
        repository: Repository[T, ID],
        cache_prefix: str,
        ttl_seconds: int = 300,
    ):
        self._repo = repository
        self._prefix = cache_prefix
        self._ttl = ttl_seconds
        self._cache = get_cache()

    def _cache_key(self, id: ID) -> str:
        return f"{self._prefix}:{id}"

    async def get(self, id: ID) -> T | None:
        cache_key = self._cache_key(id)
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached

        entity = await self._repo.get(id)
        if entity:
            from datetime import timedelta
            await self._cache.set(cache_key, entity, timedelta(seconds=self._ttl))
        return entity

    async def get_many(self, ids: list[ID]) -> list[T]:
        # Try cache first for each
        results = []
        missing_ids = []

        for id in ids:
            cached = await self._cache.get(self._cache_key(id))
            if cached:
                results.append(cached)
            else:
                missing_ids.append(id)

        # Fetch missing from repo
        if missing_ids:
            from_repo = await self._repo.get_many(missing_ids)
            for entity in from_repo:
                from datetime import timedelta
                await self._cache.set(
                    self._cache_key(getattr(entity, "id")),
                    entity,
                    timedelta(seconds=self._ttl),
                )
            results.extend(from_repo)

        return results

    async def find(self, *args, **kwargs) -> list[T]:
        # Don't cache find results (too complex to invalidate)
        return await self._repo.find(*args, **kwargs)

    async def find_one(self, filters: dict[str, Any]) -> T | None:
        return await self._repo.find_one(filters)

    async def find_by_spec(self, spec: Specification[T]) -> list[T]:
        return await self._repo.find_by_spec(spec)

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        return await self._repo.count(filters)

    async def exists(self, id: ID) -> bool:
        return await self._repo.exists(id)

    async def add(self, entity: T) -> T:
        result = await self._repo.add(entity)
        await self._cache.invalidate(self._cache_key(getattr(result, "id")))
        return result

    async def add_many(self, entities: list[T]) -> list[T]:
        results = await self._repo.add_many(entities)
        for entity in results:
            await self._cache.invalidate(self._cache_key(getattr(entity, "id")))
        return results

    async def update(self, entity: T) -> T:
        result = await self._repo.update(entity)
        await self._cache.invalidate(self._cache_key(getattr(entity, "id")))
        return result

    async def delete(self, id: ID) -> bool:
        result = await self._repo.delete(id)
        if result:
            await self._cache.invalidate(self._cache_key(id))
        return result

    async def delete_many(self, ids: list[ID]) -> int:
        count = await self._repo.delete_many(ids)
        for id in ids:
            await self._cache.invalidate(self._cache_key(id))
        return count


class UnitOfWork:
    """Unit of Work pattern for transaction management.

    Coordinates multiple repositories and ensures all changes
    are committed together or rolled back on failure.

    Example:
        async with UnitOfWork() as uow:
            user = await uow.users.get(1)
            user.name = "New Name"
            await uow.users.update(user)
            await uow.commit()
    """

    def __init__(self):
        self._repositories: dict[str, Repository] = {}
        self._new_entities: list[tuple[Repository, Any]] = []
        self._dirty_entities: list[tuple[Repository, Any]] = []
        self._deleted_ids: list[tuple[Repository, Any]] = []
        self._committed = False

    def register_repository(self, name: str, repository: Repository) -> None:
        """Register a repository with this unit of work."""
        self._repositories[name] = repository

    def __getattr__(self, name: str) -> Repository:
        """Get a repository by name."""
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._repositories:
            raise AttributeError(f"No repository named '{name}'")
        return self._repositories[name]

    def add_new(self, repository: Repository, entity: Any) -> None:
        """Mark an entity as new (to be inserted)."""
        self._new_entities.append((repository, entity))

    def add_dirty(self, repository: Repository, entity: Any) -> None:
        """Mark an entity as dirty (to be updated)."""
        self._dirty_entities.append((repository, entity))

    def add_deleted(self, repository: Repository, entity_id: Any) -> None:
        """Mark an entity as deleted."""
        self._deleted_ids.append((repository, entity_id))

    async def commit(self) -> None:
        """Commit all pending changes."""
        try:
            # Insert new entities
            for repo, entity in self._new_entities:
                await repo.add(entity)

            # Update dirty entities
            for repo, entity in self._dirty_entities:
                await repo.update(entity)

            # Delete marked entities
            for repo, entity_id in self._deleted_ids:
                await repo.delete(entity_id)

            self._committed = True
            self._clear_tracking()

            # Emit event
            await EventBus.get_instance().publish(
                Event(
                    type=EventType.AUDIT_LOG,
                    payload={
                        "action": "unit_of_work_commit",
                        "new_count": len(self._new_entities),
                        "update_count": len(self._dirty_entities),
                        "delete_count": len(self._deleted_ids),
                    },
                    source="unit_of_work",
                )
            )

        except Exception as e:
            await self.rollback()
            raise

    async def rollback(self) -> None:
        """Rollback all pending changes."""
        self._clear_tracking()

    def _clear_tracking(self) -> None:
        """Clear all tracked changes."""
        self._new_entities.clear()
        self._dirty_entities.clear()
        self._deleted_ids.clear()

    async def __aenter__(self) -> "UnitOfWork":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            await self.rollback()
        elif not self._committed:
            await self.rollback()
