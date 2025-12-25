"""Protocol definitions for dependency inversion across the codebase."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Sequence, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

T = TypeVar("T")


@runtime_checkable
class Embedder(Protocol):
    """Protocol for text embedding providers."""

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        ...

    async def embed_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 100,
    ) -> list[list[float]]:
        """Embed multiple texts in batches."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM completion providers."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """Generate a completion from messages."""
        ...


@runtime_checkable
class GraphClient(Protocol):
    """Protocol for graph database clients."""

    async def connect(self) -> None:
        """Establish connection to the database."""
        ...

    async def close(self) -> None:
        """Close the database connection."""
        ...

    async def execute(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a read query."""
        ...

    async def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a write query."""
        ...

    async def health_check(self) -> bool:
        """Check if the database is healthy."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector database clients."""

    async def connect(self) -> None:
        """Establish connection to the vector store."""
        ...

    async def close(self) -> None:
        """Close the vector store connection."""
        ...

    async def upsert(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """Insert or update vectors."""
        ...

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors."""
        ...

    async def delete(
        self,
        collection: str,
        filters: dict[str, Any],
    ) -> None:
        """Delete vectors matching filters."""
        ...

    async def health_check(self) -> bool:
        """Check if the vector store is healthy."""
        ...


@runtime_checkable
class ProgressCallback(Protocol):
    """Protocol for progress reporting."""

    def __call__(self, current: int, total: int, message: str = "") -> None:
        """Report progress."""
        ...


@runtime_checkable
class Chunker(Protocol):
    """Protocol for code chunking strategies."""

    def chunk(self, content: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        """Split content into chunks with metadata."""
        ...


@runtime_checkable
class Repository(Protocol[T]):
    """Generic repository protocol for data access."""

    async def get(self, id: str) -> T | None:
        """Get an entity by ID."""
        ...

    async def list(self) -> list[T]:
        """List all entities."""
        ...

    async def save(self, entity: T) -> None:
        """Save an entity."""
        ...

    async def delete(self, id: str) -> bool:
        """Delete an entity by ID."""
        ...
