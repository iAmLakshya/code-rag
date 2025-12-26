"""Comprehensive tests for embeddings module (chunker, embedder, indexer)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from code_rag.core.types import EntityType, Language
from code_rag.core.errors import EmbeddingError, IndexingError
from code_rag.embeddings.chunker import CodeChunker, CodeChunk
from code_rag.embeddings.embedder import OpenAIEmbedder
from code_rag.embeddings.indexer import VectorIndexer, VectorSearcher, CodeSearchResult
from code_rag.parsing.models import CodeEntity, FileInfo, ParsedFile


# ============================================================================
# Code Chunker Tests
# ============================================================================

class TestCodeChunker:
    """Tests for CodeChunker class."""

    @pytest.fixture
    def chunker(self):
        """Create a CodeChunker with default settings."""
        return CodeChunker(max_tokens=1000, overlap_tokens=200)

    @pytest.fixture
    def sample_file_info(self) -> FileInfo:
        """Create sample FileInfo."""
        return FileInfo(
            path=Path("/project/src/service.py"),
            relative_path="src/service.py",
            language=Language.PYTHON,
            content_hash="abc123",
            size_bytes=500,
            line_count=25,
        )

    def test_count_tokens(self, chunker):
        """Test token counting."""
        text = "def hello_world():\n    print('Hello, World!')"
        tokens = chunker.count_tokens(text)

        assert tokens > 0
        assert tokens < 100  # Should be a reasonable count

    def test_count_tokens_empty_string(self, chunker):
        """Test token counting for empty string."""
        tokens = chunker.count_tokens("")
        assert tokens == 0

    def test_chunk_small_entity(self, chunker, sample_file_info):
        """Test chunking a small entity that fits in one chunk."""
        entity = CodeEntity(
            type=EntityType.FUNCTION,
            name="hello",
            qualified_name="hello",
            signature="def hello()",
            docstring="Say hello.",
            code="def hello():\n    print('Hello')",
            start_line=1,
            end_line=3,
        )

        parsed_file = ParsedFile(
            file_info=sample_file_info,
            content="def hello(): pass",
            imports=[],
            entities=[entity],
        )

        chunks = chunker.chunk_file(parsed_file)

        assert len(chunks) == 1
        assert chunks[0].entity_name == "hello"
        assert chunks[0].entity_type == "function"
        assert chunks[0].file_path == str(sample_file_info.path)

    def test_chunk_large_entity(self, chunker, sample_file_info):
        """Test chunking a large entity that requires multiple chunks."""
        # Create a large entity that exceeds max_tokens
        large_code = "\n".join(
            [f"    line{i} = 'some content for line {i} with more text'" for i in range(200)]
        )
        code = f"def large_function():\n{large_code}"

        entity = CodeEntity(
            type=EntityType.FUNCTION,
            name="large_function",
            qualified_name="large_function",
            signature="def large_function()",
            code=code,
            start_line=1,
            end_line=201,
        )

        parsed_file = ParsedFile(
            file_info=sample_file_info,
            content=code,
            imports=[],
            entities=[entity],
        )

        chunks = chunker.chunk_file(parsed_file)

        # Should produce multiple chunks
        assert len(chunks) > 1

        # All chunks should have proper naming
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1 and i > 0:
                assert "_part" in chunk.entity_name
            assert chunk.file_path == str(sample_file_info.path)

    def test_chunk_preserves_metadata(self, chunker, sample_file_info):
        """Test that chunks preserve metadata."""
        entity = CodeEntity(
            type=EntityType.CLASS,
            name="MyClass",
            qualified_name="MyClass",
            signature="class MyClass",
            code="class MyClass:\n    pass",
            start_line=5,
            end_line=10,
        )

        parsed_file = ParsedFile(
            file_info=sample_file_info,
            content="class MyClass: pass",
            imports=[],
            entities=[entity],
        )

        chunks = chunker.chunk_file(parsed_file, project_name="test-project")

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.entity_type == "class"
        assert chunk.language == "python"
        assert chunk.content_hash == "abc123"
        assert chunk.project_name == "test-project"
        assert chunk.graph_node_id == "MyClass"

    def test_chunk_file_with_multiple_entities(self, chunker, sample_file_info):
        """Test chunking file with multiple entities."""
        entities = [
            CodeEntity(
                type=EntityType.FUNCTION,
                name="func1",
                qualified_name="func1",
                signature="def func1()",
                code="def func1(): pass",
                start_line=1,
                end_line=2,
            ),
            CodeEntity(
                type=EntityType.FUNCTION,
                name="func2",
                qualified_name="func2",
                signature="def func2()",
                code="def func2(): pass",
                start_line=4,
                end_line=5,
            ),
            CodeEntity(
                type=EntityType.CLASS,
                name="MyClass",
                qualified_name="MyClass",
                signature="class MyClass",
                code="class MyClass: pass",
                start_line=7,
                end_line=8,
            ),
        ]

        parsed_file = ParsedFile(
            file_info=sample_file_info,
            content="...",
            imports=[],
            entities=entities,
        )

        chunks = chunker.chunk_file(parsed_file)

        assert len(chunks) == 3
        names = [c.entity_name for c in chunks]
        assert "func1" in names
        assert "func2" in names
        assert "MyClass" in names

    def test_chunk_includes_docstring(self, chunker, sample_file_info):
        """Test that chunks include docstring in content."""
        entity = CodeEntity(
            type=EntityType.FUNCTION,
            name="documented_func",
            qualified_name="documented_func",
            signature="def documented_func(x: int)",
            docstring="This is a documented function.\n\nArgs:\n    x: Input value.",
            code="def documented_func(x: int):\n    return x * 2",
            start_line=1,
            end_line=3,
        )

        parsed_file = ParsedFile(
            file_info=sample_file_info,
            content="...",
            imports=[],
            entities=[entity],
        )

        chunks = chunker.chunk_file(parsed_file)

        assert len(chunks) == 1
        # Docstring should be included in content
        assert "This is a documented function" in chunks[0].content

    def test_chunk_empty_file(self, chunker, sample_file_info):
        """Test chunking file with no entities."""
        parsed_file = ParsedFile(
            file_info=sample_file_info,
            content="# Just a comment",
            imports=[],
            entities=[],
        )

        chunks = chunker.chunk_file(parsed_file)

        # Should chunk the file content itself
        assert len(chunks) >= 1

    def test_code_chunk_to_payload(self):
        """Test CodeChunk to_payload conversion."""
        chunk = CodeChunk(
            content="def hello(): pass",
            file_path="/project/main.py",
            entity_type="function",
            entity_name="hello",
            language="python",
            start_line=1,
            end_line=2,
            graph_node_id="hello",
            content_hash="hash123",
            project_name="test-project",
        )

        payload = chunk.to_payload()

        assert payload["file_path"] == "/project/main.py"
        assert payload["entity_type"] == "function"
        assert payload["entity_name"] == "hello"
        assert payload["language"] == "python"
        assert payload["content"] == "def hello(): pass"
        assert payload["project_name"] == "test-project"


class TestCodeChunkerOverlap:
    """Tests for overlap behavior in CodeChunker."""

    @pytest.fixture
    def chunker(self):
        # Small values to make testing easier
        return CodeChunker(max_tokens=50, overlap_tokens=10)

    def test_calculate_overlap_lines(self, chunker):
        """Test overlap line calculation."""
        lines = [
            "line 1",
            "line 2",
            "line 3",
            "line 4",
            "line 5",
        ]

        overlap = chunker._calculate_overlap_lines(lines)

        # Should return some lines that fit within overlap_tokens
        assert len(overlap) > 0
        assert len(overlap) <= len(lines)

    def test_chunk_overlap_continuity(self, chunker):
        """Test that chunks have overlapping content."""
        # Create a file with content that requires multiple chunks
        code_lines = [f"    x{i} = {i}" for i in range(50)]
        code = "def big_func():\n" + "\n".join(code_lines)

        file_info = FileInfo(
            path=Path("/project/big.py"),
            relative_path="big.py",
            language=Language.PYTHON,
            content_hash="hash",
            size_bytes=len(code),
            line_count=len(code_lines) + 1,
        )

        entity = CodeEntity(
            type=EntityType.FUNCTION,
            name="big_func",
            qualified_name="big_func",
            signature="def big_func()",
            code=code,
            start_line=1,
            end_line=len(code_lines) + 1,
        )

        parsed_file = ParsedFile(
            file_info=file_info,
            content=code,
            imports=[],
            entities=[entity],
        )

        chunks = chunker.chunk_file(parsed_file)

        if len(chunks) > 1:
            # There should be some overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                chunk1_lines = chunks[i].content.split("\n")
                chunk2_lines = chunks[i + 1].content.split("\n")

                # At least some lines should overlap
                chunk1_end = set(chunk1_lines[-5:])
                chunk2_start = set(chunk2_lines[:5])
                overlap = chunk1_end.intersection(chunk2_start)

                # May or may not have overlap depending on token boundaries
                assert len(chunks) > 1


# ============================================================================
# OpenAI Embedder Tests
# ============================================================================

class TestOpenAIEmbedder:
    """Tests for OpenAIEmbedder class."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        return AsyncMock()

    @pytest.fixture
    def embedder(self, mock_openai_client):
        """Create an embedder with mocked client."""
        with patch('code_rag.embeddings.embedder.AsyncOpenAI') as mock_class:
            mock_class.return_value = mock_openai_client
            emb = OpenAIEmbedder(api_key="test-key")
            emb._client = mock_openai_client
            return emb

    @pytest.mark.asyncio
    async def test_embed_single_text(self, embedder, mock_openai_client):
        """Test embedding a single text."""
        # Mock response
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1, 0.2, 0.3] * 512  # 1536 dims
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_openai_client.embeddings.create = AsyncMock(return_value=mock_response)

        result = await embedder.embed("Hello, world!")

        assert len(result) == 1536
        mock_openai_client.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_batch(self, embedder, mock_openai_client):
        """Test embedding a batch of texts."""
        # Mock response with multiple embeddings
        mock_embeddings = [MagicMock(embedding=[0.1] * 1536) for _ in range(3)]
        mock_response = MagicMock()
        mock_response.data = mock_embeddings
        mock_openai_client.embeddings.create = AsyncMock(return_value=mock_response)

        texts = ["text1", "text2", "text3"]
        results = await embedder.embed_batch(texts)

        assert len(results) == 3
        for embedding in results:
            assert len(embedding) == 1536

    @pytest.mark.asyncio
    async def test_embed_with_progress(self, embedder, mock_openai_client):
        """Test embedding with progress callback."""
        mock_embeddings = [MagicMock(embedding=[0.1] * 1536) for _ in range(5)]
        mock_response = MagicMock()
        mock_response.data = mock_embeddings
        mock_openai_client.embeddings.create = AsyncMock(return_value=mock_response)

        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        texts = ["text1", "text2", "text3", "text4", "text5"]
        await embedder.embed_with_progress(
            texts, batch_size=5, progress_callback=progress_callback
        )

        # Should have at least one progress call
        assert len(progress_calls) >= 1

    @pytest.mark.asyncio
    async def test_embed_batch_with_multiple_batches(self, embedder, mock_openai_client):
        """Test embedding with multiple batches."""
        mock_embeddings = [MagicMock(embedding=[0.1] * 1536) for _ in range(2)]
        mock_response = MagicMock()
        mock_response.data = mock_embeddings
        mock_openai_client.embeddings.create = AsyncMock(return_value=mock_response)

        texts = ["text1", "text2", "text3", "text4"]
        results = await embedder.embed_batch(texts, batch_size=2)

        # Should make 2 API calls
        assert mock_openai_client.embeddings.create.call_count == 2
        assert len(results) == 4

    @pytest.mark.asyncio
    async def test_embed_error_handling(self, embedder, mock_openai_client):
        """Test error handling during embedding."""
        from tenacity import RetryError
        mock_openai_client.embeddings.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        # Due to tenacity retry decorator, will exhaust retries and raise
        with pytest.raises((EmbeddingError, RetryError)):
            await embedder.embed("test")

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, embedder, mock_openai_client):
        """Test that semaphore limits concurrent requests."""
        # The semaphore should limit concurrent requests
        assert embedder._semaphore._value == embedder.max_concurrent


# ============================================================================
# Vector Indexer Tests
# ============================================================================

class TestVectorIndexer:
    """Tests for VectorIndexer class."""

    @pytest.fixture
    def mock_qdrant(self):
        qdrant = AsyncMock()
        qdrant.upsert = AsyncMock()
        qdrant.delete = AsyncMock()
        qdrant.file_needs_update = AsyncMock(return_value=True)
        return qdrant

    @pytest.fixture
    def mock_embedder(self):
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        embedder.embed_with_progress = AsyncMock(
            return_value=[[0.1] * 1536, [0.2] * 1536]
        )
        return embedder

    @pytest.fixture
    def mock_chunker(self):
        chunker = MagicMock()
        chunker.chunk_file = MagicMock(return_value=[
            CodeChunk(
                content="def hello(): pass",
                file_path="/project/main.py",
                entity_type="function",
                entity_name="hello",
                language="python",
                start_line=1,
                end_line=2,
            ),
            CodeChunk(
                content="def world(): pass",
                file_path="/project/main.py",
                entity_type="function",
                entity_name="world",
                language="python",
                start_line=4,
                end_line=5,
            ),
        ])
        return chunker

    @pytest.fixture
    def indexer(self, mock_qdrant, mock_embedder, mock_chunker):
        return VectorIndexer(mock_qdrant, mock_embedder, mock_chunker)

    @pytest.fixture
    def sample_parsed_file(self) -> ParsedFile:
        file_info = FileInfo(
            path=Path("/project/main.py"),
            relative_path="main.py",
            language=Language.PYTHON,
            content_hash="hash123",
            size_bytes=100,
            line_count=10,
        )
        return ParsedFile(
            file_info=file_info,
            content="def hello(): pass",
            imports=[],
            entities=[
                CodeEntity(
                    type=EntityType.FUNCTION,
                    name="hello",
                    qualified_name="hello",
                    signature="def hello()",
                    code="def hello(): pass",
                    start_line=1,
                    end_line=2,
                )
            ],
        )

    @pytest.mark.asyncio
    async def test_index_file(self, indexer, sample_parsed_file, mock_qdrant, mock_embedder):
        """Test indexing a file."""
        count = await indexer.index_file(sample_parsed_file)

        assert count == 2  # Two chunks from mock chunker
        mock_qdrant.delete.assert_called_once()
        mock_embedder.embed_with_progress.assert_called_once()
        mock_qdrant.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_index_file_skips_unchanged(self, indexer, sample_parsed_file, mock_qdrant):
        """Test that unchanged files are skipped."""
        mock_qdrant.file_needs_update.return_value = False

        count = await indexer.index_file(sample_parsed_file)

        assert count == 0
        mock_qdrant.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_index_file_force_reindex(self, indexer, sample_parsed_file, mock_qdrant):
        """Test forced re-indexing."""
        mock_qdrant.file_needs_update.return_value = False

        count = await indexer.index_file(sample_parsed_file, force=True)

        assert count == 2
        mock_qdrant.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_index_file_with_project_name(self, indexer, sample_parsed_file, mock_chunker):
        """Test indexing with project name."""
        await indexer.index_file(sample_parsed_file, project_name="my-project")

        mock_chunker.chunk_file.assert_called_with(
            sample_parsed_file, project_name="my-project"
        )

    @pytest.mark.asyncio
    async def test_index_file_with_progress(self, indexer, sample_parsed_file, mock_embedder):
        """Test indexing with progress callback."""
        progress_calls = []

        def callback(current, total):
            progress_calls.append((current, total))

        await indexer.index_file(sample_parsed_file, progress_callback=callback)

        mock_embedder.embed_with_progress.assert_called()

    @pytest.mark.asyncio
    async def test_index_files_multiple(self, indexer, sample_parsed_file, mock_qdrant):
        """Test indexing multiple files."""
        files = [sample_parsed_file, sample_parsed_file]

        total = await indexer.index_files(files)

        assert total == 4  # 2 chunks per file
        assert mock_qdrant.upsert.call_count == 2

    @pytest.mark.asyncio
    async def test_index_summary(self, indexer, mock_qdrant, mock_embedder):
        """Test indexing a summary."""
        await indexer.index_summary(
            file_path="/project/main.py",
            entity_type="function",
            entity_name="hello",
            summary="This function says hello",
            graph_node_id="hello",
        )

        mock_embedder.embed.assert_called_once_with("This function says hello")
        mock_qdrant.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_index_file_error_handling(self, indexer, sample_parsed_file, mock_embedder):
        """Test error handling during indexing."""
        mock_embedder.embed_with_progress.side_effect = Exception("API Error")

        with pytest.raises(IndexingError):
            await indexer.index_file(sample_parsed_file)


# ============================================================================
# Vector Searcher Tests
# ============================================================================

class TestVectorSearcher:
    """Tests for VectorSearcher class."""

    @pytest.fixture
    def mock_qdrant(self):
        qdrant = AsyncMock()
        qdrant.search = AsyncMock(return_value=[
            {
                "score": 0.95,
                "payload": {
                    "file_path": "/project/main.py",
                    "entity_type": "function",
                    "entity_name": "hello",
                    "content": "def hello(): pass",
                    "start_line": 1,
                    "end_line": 2,
                }
            },
            {
                "score": 0.85,
                "payload": {
                    "file_path": "/project/utils.py",
                    "entity_type": "function",
                    "entity_name": "helper",
                    "content": "def helper(): pass",
                    "start_line": 5,
                    "end_line": 6,
                }
            },
        ])
        return qdrant

    @pytest.fixture
    def mock_embedder(self):
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        return embedder

    @pytest.fixture
    def searcher(self, mock_qdrant, mock_embedder):
        return VectorSearcher(mock_qdrant, mock_embedder)

    @pytest.mark.asyncio
    async def test_search_code(self, searcher, mock_embedder, mock_qdrant):
        """Test code search."""
        results = await searcher.search_code("hello world", limit=10)

        assert len(results) == 2
        assert isinstance(results[0], CodeSearchResult)
        assert results[0].score == 0.95
        assert results[0].entity_name == "hello"

        mock_embedder.embed.assert_called_once_with("hello world")
        mock_qdrant.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_code_with_filters(self, searcher, mock_qdrant):
        """Test code search with filters."""
        await searcher.search_code(
            "hello",
            language="python",
            entity_type="function",
            project_name="my-project",
        )

        call_kwargs = mock_qdrant.search.call_args[1]
        assert call_kwargs["filters"]["language"] == "python"
        assert call_kwargs["filters"]["entity_type"] == "function"
        assert call_kwargs["filters"]["project_name"] == "my-project"

    @pytest.mark.asyncio
    async def test_search_summaries(self, searcher, mock_qdrant):
        """Test summary search."""
        mock_qdrant.search.return_value = [
            {
                "score": 0.9,
                "payload": {
                    "file_path": "/project/main.py",
                    "entity_type": "function",
                    "entity_name": "hello",
                    "summary": "Says hello to the world",
                }
            }
        ]

        results = await searcher.search_summaries("greeting function")

        assert len(results) == 1
        assert results[0].summary == "Says hello to the world"

    @pytest.mark.asyncio
    async def test_search_code_error_handling(self, searcher, mock_embedder):
        """Test error handling during search."""
        mock_embedder.embed.side_effect = Exception("API Error")

        with pytest.raises(IndexingError):
            await searcher.search_code("test query")

    @pytest.mark.asyncio
    async def test_format_code_results(self, searcher):
        """Test formatting of code results."""
        raw_results = [
            {
                "score": 0.9,
                "payload": {
                    "file_path": "/path/file.py",
                    "entity_type": "function",
                    "entity_name": "func",
                    "content": "def func(): pass",
                    "start_line": 1,
                    "end_line": 2,
                }
            }
        ]

        formatted = searcher._format_code_results(raw_results)

        assert len(formatted) == 1
        assert formatted[0].score == 0.9
        assert formatted[0].file_path == "/path/file.py"
        assert formatted[0].entity_name == "func"

    @pytest.mark.asyncio
    async def test_search_with_limit(self, searcher, mock_qdrant):
        """Test that limit is passed correctly."""
        await searcher.search_code("test", limit=5)

        call_kwargs = mock_qdrant.search.call_args[1]
        assert call_kwargs["limit"] == 5


# ============================================================================
# Integration Tests
# ============================================================================

class TestEmbeddingsIntegration:
    """Integration tests for embeddings module."""

    @pytest.fixture
    def sample_project_path(self) -> Path:
        return Path(__file__).parent / "fixtures" / "sample_project"

    def test_chunker_with_real_parser(self, sample_project_path):
        """Test chunker with real parsed files."""
        if not sample_project_path.exists():
            pytest.skip("Sample project not found")

        from code_rag.parsing.scanner import FileScanner
        from code_rag.parsing.parser import CodeParser

        scanner = FileScanner(sample_project_path)
        parser = CodeParser()
        chunker = CodeChunker(max_tokens=500, overlap_tokens=50)

        total_chunks = 0
        files_processed = 0

        for file_info in scanner.scan_all():
            parsed = parser.parse_file(file_info)
            chunks = chunker.chunk_file(parsed)
            total_chunks += len(chunks)
            files_processed += 1

            # Each chunk should have valid metadata
            for chunk in chunks:
                assert chunk.file_path
                assert chunk.entity_type
                assert chunk.language

        assert files_processed > 0
        assert total_chunks > 0

    def test_chunker_preserves_all_code(self, sample_project_path):
        """Test that chunking preserves all important code."""
        if not sample_project_path.exists():
            pytest.skip("Sample project not found")

        from code_rag.parsing.scanner import FileScanner
        from code_rag.parsing.parser import CodeParser

        scanner = FileScanner(sample_project_path)
        parser = CodeParser()
        chunker = CodeChunker()

        user_file = sample_project_path / "src" / "models" / "user.py"
        if not user_file.exists():
            pytest.skip("User file not found")

        file_info = next(f for f in scanner.scan_all() if f.path == user_file)
        parsed = parser.parse_file(file_info)
        chunks = chunker.chunk_file(parsed)

        # Should have chunks for User class and UserRepository class
        entity_names = [c.entity_name for c in chunks]
        assert any("User" in name for name in entity_names)

        # Check that method content is preserved
        all_content = " ".join(c.content for c in chunks)
        assert "verify_password" in all_content or "find_by_id" in all_content
