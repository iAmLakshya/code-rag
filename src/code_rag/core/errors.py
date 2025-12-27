class CodeRAGError(Exception):
    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{self.args[0]} (caused by: {self.cause})"
        return str(self.args[0])


class ConfigurationError(CodeRAGError):
    pass


class ConnectionError(CodeRAGError):
    pass


class ParsingError(CodeRAGError):
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
    pass


class VectorStoreError(CodeRAGError):
    pass


class EmbeddingError(CodeRAGError):
    pass


class IndexingError(CodeRAGError):
    def __init__(
        self,
        message: str,
        stage: str | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message, cause)
        self.stage = stage


class QueryError(CodeRAGError):
    pass


class SummarizationError(CodeRAGError):
    pass
