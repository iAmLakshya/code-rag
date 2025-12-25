"""Pydantic models for code entities."""

from pathlib import Path

from pydantic import BaseModel, Field

from code_rag.core.types import EntityType, Language


class ImportInfo(BaseModel):
    """Information about an import statement."""

    name: str = Field(description="Imported module or symbol name")
    alias: str | None = Field(default=None, description="Import alias if any")
    source: str | None = Field(default=None, description="Source module for from imports")
    is_external: bool = Field(default=True, description="Whether it's an external package")
    line_number: int = Field(description="Line number of the import")


class CodeEntity(BaseModel):
    """A code entity (function, class, method)."""

    type: EntityType = Field(description="Type of the entity")
    name: str = Field(description="Name of the entity")
    qualified_name: str = Field(description="Fully qualified name")
    signature: str | None = Field(default=None, description="Function/method signature")
    docstring: str | None = Field(default=None, description="Docstring if present")
    code: str = Field(description="Source code of the entity")
    start_line: int = Field(description="Starting line number")
    end_line: int = Field(description="Ending line number")
    is_async: bool = Field(default=False, description="Whether it's async")
    is_static: bool = Field(default=False, description="Whether it's static (for methods)")
    is_classmethod: bool = Field(default=False, description="Whether it's a classmethod")
    decorators: list[str] = Field(default_factory=list, description="Applied decorators")
    parent_class: str | None = Field(default=None, description="Parent class for methods")
    base_classes: list[str] = Field(default_factory=list, description="Base classes for classes")
    calls: list[str] = Field(default_factory=list, description="Functions/methods called")
    children: list["CodeEntity"] = Field(default_factory=list, description="Nested entities")


class FileInfo(BaseModel):
    """Information about a source file."""

    path: Path = Field(description="Absolute file path")
    relative_path: str = Field(description="Relative path from repo root")
    language: Language = Field(description="Programming language")
    content_hash: str = Field(description="SHA256 hash of file content")
    size_bytes: int = Field(description="File size in bytes")
    line_count: int = Field(description="Number of lines")


class ParsedFile(BaseModel):
    """A fully parsed source file."""

    file_info: FileInfo = Field(description="File information")
    content: str = Field(description="File content")
    imports: list[ImportInfo] = Field(default_factory=list, description="Import statements")
    entities: list[CodeEntity] = Field(default_factory=list, description="Code entities")
    summary: str | None = Field(default=None, description="AI-generated summary")

    @property
    def all_entities(self) -> list[CodeEntity]:
        """Get all entities including nested ones."""
        result = []
        stack = list(self.entities)
        while stack:
            entity = stack.pop()
            result.append(entity)
            stack.extend(entity.children)
        return result

    @property
    def functions(self) -> list[CodeEntity]:
        """Get all function entities."""
        return [e for e in self.all_entities if e.type == EntityType.FUNCTION]

    @property
    def classes(self) -> list[CodeEntity]:
        """Get all class entities."""
        return [e for e in self.all_entities if e.type == EntityType.CLASS]

    @property
    def methods(self) -> list[CodeEntity]:
        """Get all method entities."""
        return [e for e in self.all_entities if e.type == EntityType.METHOD]
