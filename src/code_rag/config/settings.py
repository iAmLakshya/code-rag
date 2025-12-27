from functools import lru_cache
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    memgraph_host: str = Field(default="localhost")
    memgraph_port: int = Field(default=7687, ge=1, le=65535)
    memgraph_user: str = Field(default="memgraph")
    memgraph_password: str = Field(default="memgraph")

    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333, ge=1, le=65535)
    qdrant_grpc_port: int = Field(default=6334, ge=1, le=65535)

    @property
    def memgraph_uri(self) -> str:
        return f"bolt://{self.memgraph_host}:{self.memgraph_port}"

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


class AISettings(BaseSettings):
    """Supports: OpenAI (default), Ollama, Anthropic, Google."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = Field(default="openai")
    embedding_provider: str = Field(default="openai")

    openai_api_key: SecretStr = Field(default=SecretStr(""))
    ollama_base_url: str = Field(default="http://localhost:11434/v1")
    anthropic_api_key: SecretStr = Field(default=SecretStr(""))
    google_api_key: SecretStr = Field(default=SecretStr(""))

    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimensions: int = Field(default=1536, gt=0)
    llm_model: str = Field(default="gpt-4o")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    @field_validator("embedding_dimensions")
    @classmethod
    def validate_embedding_dimensions(cls, v: int) -> int:
        if v > 4096:
            raise ValueError("Embedding dimensions should not exceed 4096")
        return v


class IndexingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    batch_size: int = Field(default=100, gt=0)
    max_concurrent_requests: int = Field(default=5, gt=0, le=100)
    chunk_max_tokens: int = Field(default=1000, gt=0)
    chunk_overlap_tokens: int = Field(default=200, ge=0)

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info: Any) -> int:
        if hasattr(info, "data") and "chunk_max_tokens" in info.data:
            max_tokens = info.data["chunk_max_tokens"]
            if v >= max_tokens:
                raise ValueError(
                    f"chunk_overlap_tokens ({v}) must be less than chunk_max_tokens ({max_tokens})"
                )
        return v


class FileSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    supported_extensions: list[str] = Field(
        default=[".py", ".js", ".jsx", ".ts", ".tsx"]
    )
    ignore_patterns: list[str] = Field(
        default=[
            "node_modules",
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "dist",
            "build",
            ".eggs",
            "*.egg-info",
        ]
    )

    @field_validator("supported_extensions")
    @classmethod
    def validate_extensions(cls, v: list[str]) -> list[str]:
        for ext in v:
            if not ext.startswith("."):
                raise ValueError(f"File extension must start with '.': {ext}")
        return v


class QuerySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    search_limit: int = Field(default=15, gt=0, le=100)
    max_vector_results: int = Field(default=20, gt=0, le=100)
    max_centrality_lookups: int = Field(default=10, gt=0, le=50)
    max_response_tokens: int = Field(default=1500, gt=0)
    max_explanation_tokens: int = Field(default=1000, gt=0)


class Settings(BaseSettings):
    """Composed settings with backward-compatible property access."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    ai: AISettings = Field(default_factory=AISettings)
    indexing: IndexingSettings = Field(default_factory=IndexingSettings)
    files: FileSettings = Field(default_factory=FileSettings)
    query: QuerySettings = Field(default_factory=QuerySettings)

    @property
    def memgraph_host(self) -> str:
        return self.database.memgraph_host

    @property
    def memgraph_port(self) -> int:
        return self.database.memgraph_port

    @property
    def memgraph_user(self) -> str:
        return self.database.memgraph_user

    @property
    def memgraph_password(self) -> str:
        return self.database.memgraph_password

    @property
    def memgraph_uri(self) -> str:
        return self.database.memgraph_uri

    @property
    def qdrant_host(self) -> str:
        return self.database.qdrant_host

    @property
    def qdrant_port(self) -> int:
        return self.database.qdrant_port

    @property
    def qdrant_grpc_port(self) -> int:
        return self.database.qdrant_grpc_port

    @property
    def qdrant_url(self) -> str:
        return self.database.qdrant_url

    @property
    def openai_api_key(self) -> str:
        return self.ai.openai_api_key.get_secret_value()

    @property
    def embedding_model(self) -> str:
        return self.ai.embedding_model

    @property
    def embedding_dimensions(self) -> int:
        return self.ai.embedding_dimensions

    @property
    def llm_model(self) -> str:
        return self.ai.llm_model

    @property
    def llm_temperature(self) -> float:
        return self.ai.llm_temperature

    @property
    def batch_size(self) -> int:
        return self.indexing.batch_size

    @property
    def max_concurrent_requests(self) -> int:
        return self.indexing.max_concurrent_requests

    @property
    def chunk_max_tokens(self) -> int:
        return self.indexing.chunk_max_tokens

    @property
    def chunk_overlap_tokens(self) -> int:
        return self.indexing.chunk_overlap_tokens

    @property
    def supported_extensions(self) -> list[str]:
        return self.files.supported_extensions

    @property
    def ignore_patterns(self) -> list[str]:
        return self.files.ignore_patterns


@lru_cache
def get_settings() -> Settings:
    return Settings()
