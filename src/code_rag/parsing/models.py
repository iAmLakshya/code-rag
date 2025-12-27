from pathlib import Path

from pydantic import BaseModel, Field

from code_rag.core.types import EntityType, Language


class ImportInfo(BaseModel):
    name: str
    alias: str | None = None
    source: str | None = None
    is_external: bool = True
    line_number: int


class CodeEntity(BaseModel):
    type: EntityType
    name: str
    qualified_name: str
    signature: str | None = None
    docstring: str | None = None
    code: str
    start_line: int
    end_line: int
    is_async: bool = False
    is_static: bool = False
    is_classmethod: bool = False
    decorators: list[str] = Field(default_factory=list)
    parent_class: str | None = None
    base_classes: list[str] = Field(default_factory=list)
    calls: list[str] = Field(default_factory=list)
    children: list["CodeEntity"] = Field(default_factory=list)


class FileInfo(BaseModel):
    path: Path
    relative_path: str
    language: Language
    content_hash: str
    size_bytes: int
    line_count: int


class ParsedFile(BaseModel):
    file_info: FileInfo
    content: str
    imports: list[ImportInfo] = Field(default_factory=list)
    entities: list[CodeEntity] = Field(default_factory=list)
    summary: str | None = None

    @property
    def all_entities(self) -> list[CodeEntity]:
        result = []
        stack = list(self.entities)
        while stack:
            entity = stack.pop()
            result.append(entity)
            stack.extend(entity.children)
        return result

    @property
    def functions(self) -> list[CodeEntity]:
        return [e for e in self.all_entities if e.type == EntityType.FUNCTION]

    @property
    def classes(self) -> list[CodeEntity]:
        return [e for e in self.all_entities if e.type == EntityType.CLASS]

    @property
    def methods(self) -> list[CodeEntity]:
        return [e for e in self.all_entities if e.type == EntityType.METHOD]
