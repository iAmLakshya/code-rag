"""Custom exception hierarchy for code-rag."""


class CodeRAGError(Exception):
    """Base exception for all code-rag errors."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{self.args[0]} (caused by: {self.cause})"
        return str(self.args[0])


class ConfigurationError(CodeRAGError):
    """Raised when there's a configuration problem."""

    pass


class ConnectionError(CodeRAGError):
    """Raised when a connection to an external service fails."""

    pass


class ParsingError(CodeRAGError):
    """Raised when code parsing fails."""

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        line: int | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message, cause)
        self.file_path = file_path
        self.line = line


class GraphError(CodeRAGError):
    """Raised when a graph database operation fails."""

    pass


class VectorStoreError(CodeRAGError):
    """Raised when a vector store operation fails."""

    pass


class EmbeddingError(CodeRAGError):
    """Raised when embedding generation fails."""

    pass


class IndexingError(CodeRAGError):
    """Raised when indexing operations fail."""

    def __init__(
        self,
        message: str,
        stage: str | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message, cause)
        self.stage = stage


class QueryError(CodeRAGError):
    """Raised when query execution fails."""

    pass


class SummarizationError(CodeRAGError):
    """Raised when code summarization fails."""

    pass
