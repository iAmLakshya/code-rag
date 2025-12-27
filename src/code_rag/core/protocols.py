from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: Sequence[str], batch_size: int = 100) -> list[list[float]]: ...


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str: ...


@runtime_checkable
class GraphClient(Protocol):
    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def execute(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...
    async def execute_write(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...
    async def health_check(self) -> bool: ...


@runtime_checkable
class VectorStore(Protocol):
    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def upsert(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None: ...
    async def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...
    async def delete(self, collection: str, filters: dict[str, Any]) -> None: ...
    async def health_check(self) -> bool: ...


@runtime_checkable
class ProgressCallback(Protocol):
    def __call__(self, current: int, total: int, message: str = "") -> None: ...


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, content: str, metadata: dict[str, Any]) -> list[dict[str, Any]]: ...


@runtime_checkable
class Repository(Protocol[T]):
    async def get(self, id: str) -> T | None: ...
    async def list(self) -> list[T]: ...
    async def save(self, entity: T) -> None: ...
    async def delete(self, id: str) -> bool: ...
