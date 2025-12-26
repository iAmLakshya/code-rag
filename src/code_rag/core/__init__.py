"""Core abstractions and shared utilities for code-rag."""

from code_rag.core.cache import (
    ASTCache,
    BoundedCache,
    FunctionRegistry,
)
from code_rag.core.protocols import (
    Embedder,
    GraphClient,
    LLMProvider,
    VectorStore,
)
from code_rag.core.types import (
    EntityType,
    Language,
    QueryType,
    ResultSource,
)
from code_rag.core.errors import (
    CodeRAGError,
    ConfigurationError,
    ConnectionError,
    EmbeddingError,
    GraphError,
    IndexingError,
    ParsingError,
    QueryError,
    VectorStoreError,
)

__all__ = [
    "ASTCache",
    "BoundedCache",
    "Embedder",
    "FunctionRegistry",
    "GraphClient",
    "LLMProvider",
    "VectorStore",
    "EntityType",
    "Language",
    "QueryType",
    "ResultSource",
    "CodeRAGError",
    "ConfigurationError",
    "ConnectionError",
    "EmbeddingError",
    "GraphError",
    "IndexingError",
    "ParsingError",
    "QueryError",
    "VectorStoreError",
]
