"""Application settings using Pydantic Settings.

Refactored to follow SOLID principles with domain-separated configuration classes.
"""

from functools import lru_cache
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Graph and vector database configuration.

    Handles Memgraph (graph database) and Qdrant (vector database) connection settings.
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Memgraph Configuration
    memgraph_host: str = Field(
        default="localhost",
        description="Memgraph host address",
    )
    memgraph_port: int = Field(
        default=7687,
        description="Memgraph Bolt port",
        ge=1,
        le=65535,
    )
    memgraph_user: str = Field(
        default="memgraph",
        description="Memgraph username",
    )
    memgraph_password: str = Field(
        default="memgraph",
        description="Memgraph password",
    )

    # Qdrant Configuration
    qdrant_host: str = Field(
        default="localhost",
        description="Qdrant host address",
    )
    qdrant_port: int = Field(
        default=6333,
        description="Qdrant REST API port",
        ge=1,
        le=65535,
    )
    qdrant_grpc_port: int = Field(
        default=6334,
        description="Qdrant gRPC port",
        ge=1,
        le=65535,
    )

    @property
    def memgraph_uri(self) -> str:
        """Get Memgraph connection URI.

        Returns:
            Bolt protocol URI for Memgraph connection.
        """
        return f"bolt://{self.memgraph_host}:{self.memgraph_port}"

    @property
    def qdrant_url(self) -> str:
        """Get Qdrant REST API URL.

        Returns:
            HTTP URL for Qdrant REST API.
        """
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


class AISettings(BaseSettings):
    """AI provider configuration.

    Supports multiple LLM and embedding providers:
    - OpenAI (default)
    - Ollama (local models)
    - Anthropic (Claude)
    - Google (Gemini)

    Set LLM_PROVIDER and EMBEDDING_PROVIDER to switch providers.
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider Selection
    llm_provider: str = Field(
        default="openai",
        description="LLM provider: openai, ollama, anthropic, google",
    )
    embedding_provider: str = Field(
        default="openai",
        description="Embedding provider: openai, ollama, google",
    )

    # OpenAI Configuration
    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="OpenAI API key (stored securely)",
    )

    # Ollama Configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Ollama API base URL",
    )

    # Anthropic Configuration
    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Anthropic API key (stored securely)",
    )

    # Google Configuration
    google_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Google API key (stored securely)",
    )

    # Model Configuration
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name",
    )
    embedding_dimensions: int = Field(
        default=1536,
        description="Embedding vector dimensions",
        gt=0,
    )
    llm_model: str = Field(
        default="gpt-4o",
        description="LLM model for summaries and responses",
    )
    llm_temperature: float = Field(
        default=0.1,
        description="LLM temperature for generation",
        ge=0.0,
        le=2.0,
    )

    @field_validator("embedding_dimensions")
    @classmethod
    def validate_embedding_dimensions(cls, v: int) -> int:
        """Validate embedding dimensions are reasonable.

        Args:
            v: The embedding dimensions value.

        Returns:
            The validated embedding dimensions.

        Raises:
            ValueError: If dimensions are not in expected range.
        """
        if v > 4096:
            raise ValueError("Embedding dimensions should not exceed 4096")
        return v


class IndexingSettings(BaseSettings):
    """Indexing and chunking configuration.

    Handles batch processing, concurrency, and code chunking parameters.
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    batch_size: int = Field(
        default=100,
        description="Batch size for database operations",
        gt=0,
    )
    max_concurrent_requests: int = Field(
        default=5,
        description="Max concurrent API requests",
        gt=0,
        le=100,
    )
    chunk_max_tokens: int = Field(
        default=1000,
        description="Max tokens per code chunk",
        gt=0,
    )
    chunk_overlap_tokens: int = Field(
        default=200,
        description="Token overlap between chunks",
        ge=0,
    )

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info: Any) -> int:
        """Validate chunk overlap is less than max tokens.

        Args:
            v: The chunk overlap value.
            info: Validation info containing other field values.

        Returns:
            The validated chunk overlap.

        Raises:
            ValueError: If overlap is greater than or equal to max tokens.
        """
        if hasattr(info, "data") and "chunk_max_tokens" in info.data:
            max_tokens = info.data["chunk_max_tokens"]
            if v >= max_tokens:
                raise ValueError(
                    f"chunk_overlap_tokens ({v}) must be less than chunk_max_tokens ({max_tokens})"
                )
        return v


class FileSettings(BaseSettings):
    """File scanning configuration.

    Handles which file types to index and which directories to ignore.
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    supported_extensions: list[str] = Field(
        default=[".py", ".js", ".jsx", ".ts", ".tsx"],
        description="File extensions to index",
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
        ],
        description="Directory patterns to ignore during scanning",
    )

    @field_validator("supported_extensions")
    @classmethod
    def validate_extensions(cls, v: list[str]) -> list[str]:
        """Validate file extensions start with a dot.

        Args:
            v: List of file extensions.

        Returns:
            The validated list of extensions.

        Raises:
            ValueError: If any extension doesn't start with a dot.
        """
        for ext in v:
            if not ext.startswith("."):
                raise ValueError(f"File extension must start with '.': {ext}")
        return v


class Settings(BaseSettings):
    """Composed application settings.

    This class composes all domain-specific settings into a single unified
    configuration object. It maintains backward compatibility by exposing
    all settings at the root level while also organizing them by domain.

    Usage:
        settings = get_settings()

        # Access via domain-specific sections (recommended)
        api_key = settings.ai.openai_api_key
        db_uri = settings.database.memgraph_uri

        # Access via root level (backward compatible)
        api_key = settings.openai_api_key
        db_uri = settings.memgraph_uri
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Domain-specific settings sections
    database: DatabaseSettings = Field(
        default_factory=DatabaseSettings,
        description="Database configuration (Memgraph and Qdrant)",
    )
    ai: AISettings = Field(
        default_factory=AISettings,
        description="AI/ML configuration (OpenAI)",
    )
    indexing: IndexingSettings = Field(
        default_factory=IndexingSettings,
        description="Indexing and chunking configuration",
    )
    files: FileSettings = Field(
        default_factory=FileSettings,
        description="File scanning configuration",
    )

    # Backward compatibility properties - delegate to domain-specific settings

    # Database properties
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

    # AI properties
    @property
    def openai_api_key(self) -> str:
        """Backward compatible access to openai_api_key.

        Returns the API key as a plain string for compatibility.
        """
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

    # Indexing properties
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

    # File properties
    @property
    def supported_extensions(self) -> list[str]:
        return self.files.supported_extensions

    @property
    def ignore_patterns(self) -> list[str]:
        return self.files.ignore_patterns


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    This function uses LRU cache to ensure only one Settings instance
    is created and reused throughout the application lifecycle.

    Returns:
        Singleton Settings instance with all configuration loaded.
    """
    return Settings()
