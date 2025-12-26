"""Tests for the refactored config module."""

import pytest
from pydantic import SecretStr, ValidationError

from code_rag.config import (
    AISettings,
    DatabaseSettings,
    FileSettings,
    IndexingSettings,
    Settings,
    get_settings,
)


class TestDatabaseSettings:
    """Tests for DatabaseSettings class."""

    def test_default_values(self):
        """Test default database configuration values."""
        db = DatabaseSettings()

        assert db.memgraph_host == "localhost"
        assert db.memgraph_port == 7687
        assert db.memgraph_user == "memgraph"
        assert db.memgraph_password == "memgraph"
        assert db.qdrant_host == "localhost"
        assert db.qdrant_port == 6333
        assert db.qdrant_grpc_port == 6334

    def test_memgraph_uri_computation(self):
        """Test Memgraph URI is computed correctly."""
        db = DatabaseSettings(memgraph_host="test-host", memgraph_port=7688)

        assert db.memgraph_uri == "bolt://test-host:7688"

    def test_qdrant_url_computation(self):
        """Test Qdrant URL is computed correctly."""
        db = DatabaseSettings(qdrant_host="vector-host", qdrant_port=6334)

        assert db.qdrant_url == "http://vector-host:6334"

    def test_port_validation(self):
        """Test port range validation."""
        # Valid ports
        db = DatabaseSettings(memgraph_port=1)
        assert db.memgraph_port == 1

        db = DatabaseSettings(memgraph_port=65535)
        assert db.memgraph_port == 65535

        # Invalid port - too high
        with pytest.raises(ValidationError) as exc_info:
            DatabaseSettings(memgraph_port=70000)
        assert "less than or equal to 65535" in str(exc_info.value)

        # Invalid port - too low
        with pytest.raises(ValidationError) as exc_info:
            DatabaseSettings(memgraph_port=0)
        assert "greater than or equal to 1" in str(exc_info.value)


class TestAISettings:
    """Tests for AISettings class."""

    def test_default_values(self):
        """Test default AI configuration values."""
        ai = AISettings()

        assert ai.embedding_model == "text-embedding-3-small"
        assert ai.embedding_dimensions == 1536
        assert ai.llm_model == "gpt-4o"
        assert ai.llm_temperature == 0.1

    def test_api_key_is_secret_str(self):
        """Test API key is stored as SecretStr."""
        ai = AISettings(openai_api_key="test-secret-key")

        assert isinstance(ai.openai_api_key, SecretStr)
        assert ai.openai_api_key.get_secret_value() == "test-secret-key"
        assert "test-secret-key" not in str(ai.openai_api_key)

    def test_temperature_validation(self):
        """Test temperature range validation."""
        # Valid temperatures
        ai = AISettings(llm_temperature=0.0)
        assert ai.llm_temperature == 0.0

        ai = AISettings(llm_temperature=2.0)
        assert ai.llm_temperature == 2.0

        # Invalid temperature - too high
        with pytest.raises(ValidationError) as exc_info:
            AISettings(llm_temperature=3.0)
        assert "less than or equal to 2" in str(exc_info.value)

        # Invalid temperature - negative
        with pytest.raises(ValidationError) as exc_info:
            AISettings(llm_temperature=-0.1)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_embedding_dimensions_validation(self):
        """Test embedding dimensions validation."""
        # Valid dimensions
        ai = AISettings(embedding_dimensions=1536)
        assert ai.embedding_dimensions == 1536

        # Invalid - too large
        with pytest.raises(ValidationError) as exc_info:
            AISettings(embedding_dimensions=5000)
        assert "should not exceed 4096" in str(exc_info.value)

        # Invalid - zero or negative
        with pytest.raises(ValidationError) as exc_info:
            AISettings(embedding_dimensions=0)
        assert "greater than 0" in str(exc_info.value)


class TestIndexingSettings:
    """Tests for IndexingSettings class."""

    def test_default_values(self):
        """Test default indexing configuration values."""
        indexing = IndexingSettings()

        assert indexing.batch_size == 100
        assert indexing.max_concurrent_requests == 5
        assert indexing.chunk_max_tokens == 1000
        assert indexing.chunk_overlap_tokens == 200

    def test_positive_number_validation(self):
        """Test validation for positive numbers."""
        # Valid values
        indexing = IndexingSettings(batch_size=1)
        assert indexing.batch_size == 1

        # Invalid - zero
        with pytest.raises(ValidationError) as exc_info:
            IndexingSettings(batch_size=0)
        assert "greater than 0" in str(exc_info.value)

    def test_max_concurrent_validation(self):
        """Test max concurrent requests validation."""
        # Valid value
        indexing = IndexingSettings(max_concurrent_requests=50)
        assert indexing.max_concurrent_requests == 50

        # Invalid - too high
        with pytest.raises(ValidationError) as exc_info:
            IndexingSettings(max_concurrent_requests=101)
        assert "less than or equal to 100" in str(exc_info.value)


class TestFileSettings:
    """Tests for FileSettings class."""

    def test_default_values(self):
        """Test default file configuration values."""
        files = FileSettings()

        assert ".py" in files.supported_extensions
        assert ".js" in files.supported_extensions
        assert "node_modules" in files.ignore_patterns
        assert "__pycache__" in files.ignore_patterns

    def test_extension_validation(self):
        """Test file extension validation."""
        # Valid extensions
        files = FileSettings(supported_extensions=[".py", ".js", ".ts"])
        assert files.supported_extensions == [".py", ".js", ".ts"]

        # Invalid - missing dot
        with pytest.raises(ValidationError) as exc_info:
            FileSettings(supported_extensions=["py", ".js"])
        assert "must start with '.'" in str(exc_info.value)


class TestSettings:
    """Tests for composed Settings class."""

    def test_domain_sections_exist(self):
        """Test that all domain sections are present."""
        settings = Settings()

        assert isinstance(settings.database, DatabaseSettings)
        assert isinstance(settings.ai, AISettings)
        assert isinstance(settings.indexing, IndexingSettings)
        assert isinstance(settings.files, FileSettings)

    def test_backward_compatibility_database(self):
        """Test backward compatible access to database settings."""
        settings = Settings()

        assert settings.memgraph_host == settings.database.memgraph_host
        assert settings.memgraph_port == settings.database.memgraph_port
        assert settings.memgraph_user == settings.database.memgraph_user
        assert settings.memgraph_password == settings.database.memgraph_password
        assert settings.memgraph_uri == settings.database.memgraph_uri
        assert settings.qdrant_host == settings.database.qdrant_host
        assert settings.qdrant_port == settings.database.qdrant_port
        assert settings.qdrant_grpc_port == settings.database.qdrant_grpc_port
        assert settings.qdrant_url == settings.database.qdrant_url

    def test_backward_compatibility_ai(self):
        """Test backward compatible access to AI settings."""
        settings = Settings()

        # API key should be plain string for backward compatibility
        assert isinstance(settings.openai_api_key, str)
        assert (
            settings.openai_api_key == settings.ai.openai_api_key.get_secret_value()
        )
        assert settings.embedding_model == settings.ai.embedding_model
        assert settings.embedding_dimensions == settings.ai.embedding_dimensions
        assert settings.llm_model == settings.ai.llm_model
        assert settings.llm_temperature == settings.ai.llm_temperature

    def test_backward_compatibility_indexing(self):
        """Test backward compatible access to indexing settings."""
        settings = Settings()

        assert settings.batch_size == settings.indexing.batch_size
        assert (
            settings.max_concurrent_requests
            == settings.indexing.max_concurrent_requests
        )
        assert settings.chunk_max_tokens == settings.indexing.chunk_max_tokens
        assert settings.chunk_overlap_tokens == settings.indexing.chunk_overlap_tokens

    def test_backward_compatibility_files(self):
        """Test backward compatible access to file settings."""
        settings = Settings()

        assert settings.supported_extensions == settings.files.supported_extensions
        assert settings.ignore_patterns == settings.files.ignore_patterns


class TestGetSettings:
    """Tests for get_settings singleton function."""

    def test_singleton_pattern(self):
        """Test that get_settings returns the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_returns_settings_instance(self):
        """Test that get_settings returns a Settings instance."""
        settings = get_settings()

        assert isinstance(settings, Settings)

    def test_cache_clear(self):
        """Test that cache can be cleared."""
        settings1 = get_settings()
        get_settings.cache_clear()
        settings2 = get_settings()

        # After cache clear, should get a new instance
        # (though values should be the same)
        assert isinstance(settings2, Settings)


class TestRealWorldUsage:
    """Tests simulating real-world usage patterns from the codebase."""

    def test_graph_client_pattern(self):
        """Test usage pattern from graph client."""
        settings = get_settings()

        # Graph client accesses these
        uri = settings.memgraph_uri
        user = settings.memgraph_user
        password = settings.memgraph_password

        assert uri.startswith("bolt://")
        assert isinstance(user, str)
        assert isinstance(password, str)

    def test_embedder_pattern(self):
        """Test usage pattern from embedder."""
        settings = get_settings()

        # Embedder accesses these
        model = settings.embedding_model
        max_concurrent = settings.max_concurrent_requests
        api_key = settings.openai_api_key

        assert isinstance(model, str)
        assert isinstance(max_concurrent, int)
        assert isinstance(api_key, str)

    def test_qdrant_client_pattern(self):
        """Test usage pattern from Qdrant client."""
        settings = get_settings()

        # Qdrant client accesses these
        host = settings.qdrant_host
        port = settings.qdrant_port
        grpc_port = settings.qdrant_grpc_port
        dimensions = settings.embedding_dimensions

        assert isinstance(host, str)
        assert isinstance(port, int)
        assert isinstance(grpc_port, int)
        assert isinstance(dimensions, int)

    def test_chunker_pattern(self):
        """Test usage pattern from chunker."""
        settings = get_settings()

        # Chunker accesses these
        max_tokens = settings.chunk_max_tokens
        overlap_tokens = settings.chunk_overlap_tokens

        assert isinstance(max_tokens, int)
        assert isinstance(overlap_tokens, int)
        assert overlap_tokens < max_tokens

    def test_scanner_pattern(self):
        """Test usage pattern from file scanner."""
        settings = get_settings()

        # Scanner accesses these
        extensions = settings.supported_extensions
        ignore_patterns = settings.ignore_patterns

        assert isinstance(extensions, list)
        assert isinstance(ignore_patterns, list)
        assert all(ext.startswith(".") for ext in extensions)

    def test_summarizer_pattern(self):
        """Test usage pattern from summarizer."""
        settings = get_settings()

        # Summarizer accesses these
        model = settings.llm_model
        temperature = settings.llm_temperature
        max_concurrent = settings.max_concurrent_requests
        api_key = settings.openai_api_key

        assert isinstance(model, str)
        assert isinstance(temperature, float)
        assert isinstance(max_concurrent, int)
        assert isinstance(api_key, str)
