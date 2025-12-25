"""Data models for project management."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ProjectIndex:
    """Represents an indexed path within a project."""

    path: str
    file_count: int = 0
    entity_count: int = 0
    chunk_count: int = 0
    indexed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("path cannot be empty")
        if self.file_count < 0:
            raise ValueError("file_count must be non-negative")
        if self.entity_count < 0:
            raise ValueError("entity_count must be non-negative")
        if self.chunk_count < 0:
            raise ValueError("chunk_count must be non-negative")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProjectIndex):
            return NotImplemented
        return (
            self.path == other.path
            and self.file_count == other.file_count
            and self.entity_count == other.entity_count
            and self.chunk_count == other.chunk_count
            and self.indexed_at == other.indexed_at
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.path,
                self.file_count,
                self.entity_count,
                self.chunk_count,
                self.indexed_at,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "file_count": self.file_count,
            "entity_count": self.entity_count,
            "chunk_count": self.chunk_count,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectIndex":
        indexed_at = data.get("indexed_at")
        if indexed_at and isinstance(indexed_at, str):
            indexed_at = datetime.fromisoformat(indexed_at)

        return cls(
            path=data["path"],
            file_count=data.get("file_count", 0),
            entity_count=data.get("entity_count", 0),
            chunk_count=data.get("chunk_count", 0),
            indexed_at=indexed_at,
        )


@dataclass(frozen=True)
class Project:
    """Represents a Code RAG project."""

    name: str
    created_at: datetime = field(default_factory=datetime.now)
    last_indexed_at: datetime | None = None
    indexes: tuple[ProjectIndex, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name cannot be empty")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Project):
            return NotImplemented
        return (
            self.name == other.name
            and self.created_at == other.created_at
            and self.last_indexed_at == other.last_indexed_at
            and self.indexes == other.indexes
        )

    def __hash__(self) -> int:
        return hash((self.name, self.created_at, self.last_indexed_at, self.indexes))

    @property
    def total_files(self) -> int:
        return sum(idx.file_count for idx in self.indexes)

    @property
    def total_entities(self) -> int:
        return sum(idx.entity_count for idx in self.indexes)

    @property
    def total_chunks(self) -> int:
        return sum(idx.chunk_count for idx in self.indexes)

    @property
    def index_paths(self) -> list[str]:
        return [idx.path for idx in self.indexes]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "last_indexed_at": (
                self.last_indexed_at.isoformat() if self.last_indexed_at else None
            ),
            "indexes": [idx.to_dict() for idx in self.indexes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Project":
        created_at = data.get("created_at")
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        last_indexed_at = data.get("last_indexed_at")
        if last_indexed_at and isinstance(last_indexed_at, str):
            last_indexed_at = datetime.fromisoformat(last_indexed_at)

        indexes = tuple(
            ProjectIndex.from_dict(idx) for idx in data.get("indexes", [])
        )

        return cls(
            name=data["name"],
            created_at=created_at or datetime.now(),
            last_indexed_at=last_indexed_at,
            indexes=indexes,
        )
