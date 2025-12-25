"""Shared type definitions and enums."""

from enum import Enum


class Language(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JSX = "jsx"
    TSX = "tsx"

    @classmethod
    def from_extension(cls, ext: str) -> "Language | None":
        """Get language from file extension."""
        mapping = {
            ".py": cls.PYTHON,
            ".js": cls.JAVASCRIPT,
            ".jsx": cls.JSX,
            ".ts": cls.TYPESCRIPT,
            ".tsx": cls.TSX,
        }
        return mapping.get(ext.lower())

    @property
    def extensions(self) -> list[str]:
        """Get file extensions for this language."""
        mapping = {
            self.PYTHON: [".py"],
            self.JAVASCRIPT: [".js"],
            self.JSX: [".jsx"],
            self.TYPESCRIPT: [".ts"],
            self.TSX: [".tsx"],
        }
        return mapping.get(self, [])


class EntityType(str, Enum):
    """Types of code entities."""

    FILE = "file"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    IMPORT = "import"
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"


class QueryType(str, Enum):
    """Types of code queries."""

    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    NAVIGATIONAL = "navigational"
    EXPLANATORY = "explanatory"


class ResultSource(str, Enum):
    """Source of search results."""

    GRAPH = "graph"
    VECTOR = "vector"
    HYBRID = "hybrid"


class PipelineStage(str, Enum):
    """Stages in the indexing pipeline."""

    SCANNING = "scanning"
    PARSING = "parsing"
    GRAPH_BUILDING = "graph_building"
    SUMMARIZING = "summarizing"
    EMBEDDING = "embedding"
    COMPLETED = "completed"
    FAILED = "failed"
