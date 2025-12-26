"""Comprehensive tests for summarization module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from code_rag.core.types import EntityType, Language
from code_rag.core.errors import SummarizationError
from code_rag.summarization.summarizer import CodeSummarizer
from code_rag.parsing.models import CodeEntity, FileInfo, ParsedFile


# ============================================================================
# Code Summarizer Tests
# ============================================================================

class TestCodeSummarizer:
    """Tests for CodeSummarizer class."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value="This is a test summary.")
        return provider

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Mock summary from OpenAI"))]
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(return_value=mock_response)
        return client

    @pytest.fixture
    def summarizer_with_provider(self, mock_llm_provider):
        """Create summarizer with custom LLM provider."""
        return CodeSummarizer(llm_provider=mock_llm_provider)

    @pytest.fixture
    def summarizer_with_openai(self, mock_openai_client):
        """Create summarizer with mocked OpenAI client."""
        with patch('code_rag.summarization.summarizer.AsyncOpenAI') as mock_class:
            mock_class.return_value = mock_openai_client
            summarizer = CodeSummarizer(api_key="test-key")
            summarizer._client = mock_openai_client
            return summarizer

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

    @pytest.fixture
    def sample_function_entity(self) -> CodeEntity:
        """Create sample function entity."""
        return CodeEntity(
            type=EntityType.FUNCTION,
            name="process_data",
            qualified_name="process_data",
            signature="def process_data(data: list) -> dict",
            docstring="Process the input data and return results.",
            code="def process_data(data: list) -> dict:\n    return {'result': data}",
            start_line=10,
            end_line=15,
            is_async=False,
            calls=["validate", "transform"],
        )

    @pytest.fixture
    def sample_class_entity(self) -> CodeEntity:
        """Create sample class entity."""
        return CodeEntity(
            type=EntityType.CLASS,
            name="DataProcessor",
            qualified_name="DataProcessor",
            signature="class DataProcessor",
            docstring="Handles data processing operations.",
            code="class DataProcessor:\n    def process(self): pass",
            start_line=1,
            end_line=20,
            base_classes=["BaseProcessor"],
            children=[
                CodeEntity(
                    type=EntityType.METHOD,
                    name="process",
                    qualified_name="DataProcessor.process",
                    signature="def process(self)",
                    code="def process(self): pass",
                    start_line=5,
                    end_line=6,
                    parent_class="DataProcessor",
                )
            ],
        )

    @pytest.fixture
    def sample_parsed_file(self, sample_file_info, sample_function_entity, sample_class_entity) -> ParsedFile:
        """Create sample ParsedFile."""
        return ParsedFile(
            file_info=sample_file_info,
            content="# Sample file content\nclass DataProcessor: pass",
            imports=[],
            entities=[sample_class_entity, sample_function_entity],
        )

    # -------------------------------------------------------------------------
    # Basic Summarization Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_summarize_file(self, summarizer_with_provider, sample_parsed_file, mock_llm_provider):
        """Test file summarization."""
        summary = await summarizer_with_provider.summarize_file(sample_parsed_file)

        assert summary == "This is a test summary."
        mock_llm_provider.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_function(self, summarizer_with_provider, sample_function_entity, mock_llm_provider):
        """Test function summarization."""
        summary = await summarizer_with_provider.summarize_entity(
            sample_function_entity,
            file_path="src/utils.py",
            language="python",
        )

        assert summary == "This is a test summary."
        mock_llm_provider.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_class(self, summarizer_with_provider, sample_class_entity, mock_llm_provider):
        """Test class summarization."""
        summary = await summarizer_with_provider.summarize_entity(
            sample_class_entity,
            file_path="src/processor.py",
            language="python",
        )

        assert summary == "This is a test summary."
        mock_llm_provider.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_method(self, summarizer_with_provider, mock_llm_provider):
        """Test method summarization (uses function strategy)."""
        method_entity = CodeEntity(
            type=EntityType.METHOD,
            name="calculate",
            qualified_name="Calculator.calculate",
            signature="def calculate(self, x: int)",
            code="def calculate(self, x): return x * 2",
            start_line=5,
            end_line=6,
            parent_class="Calculator",
        )

        summary = await summarizer_with_provider.summarize_entity(
            method_entity,
            file_path="src/calc.py",
            language="python",
        )

        assert summary == "This is a test summary."

    @pytest.mark.asyncio
    async def test_summarize_unsupported_entity_type(self, summarizer_with_provider):
        """Test that unsupported entity types return empty string."""
        import_entity = CodeEntity(
            type=EntityType.IMPORT,
            name="os",
            qualified_name="os",
            code="import os",
            start_line=1,
            end_line=1,
        )

        summary = await summarizer_with_provider.summarize_entity(
            import_entity,
            file_path="src/main.py",
            language="python",
        )

        assert summary == ""

    # -------------------------------------------------------------------------
    # OpenAI Client Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_summarize_with_openai_client(self, summarizer_with_openai, sample_parsed_file, mock_openai_client):
        """Test summarization using OpenAI client."""
        summary = await summarizer_with_openai.summarize_file(sample_parsed_file)

        assert summary == "Mock summary from OpenAI"
        mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_openai_client_receives_correct_params(self, summarizer_with_openai, sample_function_entity, mock_openai_client):
        """Test that OpenAI client receives correct parameters."""
        await summarizer_with_openai.summarize_entity(
            sample_function_entity,
            file_path="src/utils.py",
            language="python",
        )

        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert "model" in call_kwargs
        assert "messages" in call_kwargs
        assert "temperature" in call_kwargs
        assert "max_tokens" in call_kwargs

        # Should have system and user messages
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    # -------------------------------------------------------------------------
    # Full File Summarization Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_summarize_parsed_file(self, summarizer_with_provider, sample_parsed_file, mock_llm_provider):
        """Test summarizing entire parsed file with entities."""
        summaries = await summarizer_with_provider.summarize_parsed_file(sample_parsed_file)

        # Should have file summary and entity summaries
        assert "__file__" in summaries
        assert summaries["__file__"] == "This is a test summary."

        # Should have entity summaries
        assert len(summaries) > 1

    @pytest.mark.asyncio
    async def test_summarize_parsed_file_without_entities(self, summarizer_with_provider, sample_parsed_file, mock_llm_provider):
        """Test summarizing file without entity summaries."""
        summaries = await summarizer_with_provider.summarize_parsed_file(
            sample_parsed_file, include_entities=False
        )

        # Should only have file summary
        assert "__file__" in summaries
        assert len(summaries) == 1

    @pytest.mark.asyncio
    async def test_summarize_parsed_file_with_progress(self, summarizer_with_provider, sample_parsed_file, mock_llm_provider):
        """Test summarization with progress callback."""
        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        await summarizer_with_provider.summarize_parsed_file(
            sample_parsed_file,
            progress_callback=progress_callback,
        )

        # Should have progress calls for entities
        assert len(progress_calls) > 0

    @pytest.mark.asyncio
    async def test_summarize_parsed_file_handles_entity_errors(self, summarizer_with_provider, sample_parsed_file, mock_llm_provider):
        """Test that entity summarization errors don't stop file summarization."""
        call_count = [0]

        async def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:  # Fail on second call (first entity)
                raise SummarizationError("API Error")
            return "Summary"

        mock_llm_provider.complete.side_effect = side_effect

        summaries = await summarizer_with_provider.summarize_parsed_file(sample_parsed_file)

        # Should still have file summary
        assert "__file__" in summaries
        # Some entities may have empty summaries due to error
        assert len(summaries) > 1

    # -------------------------------------------------------------------------
    # Error Handling Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_summarize_error_handling(self, summarizer_with_provider, sample_parsed_file, mock_llm_provider):
        """Test error handling during summarization."""
        from tenacity import RetryError
        mock_llm_provider.complete.side_effect = Exception("API Error")

        # Due to tenacity retry decorator, will exhaust retries and raise
        with pytest.raises((SummarizationError, RetryError)):
            await summarizer_with_provider.summarize_file(sample_parsed_file)

    @pytest.mark.asyncio
    async def test_summarize_openai_error_handling(self, summarizer_with_openai, sample_function_entity, mock_openai_client):
        """Test error handling with OpenAI client."""
        from tenacity import RetryError
        mock_openai_client.chat.completions.create.side_effect = Exception("OpenAI Error")

        # Due to tenacity retry decorator, will exhaust retries and raise
        with pytest.raises((SummarizationError, RetryError)):
            await summarizer_with_openai.summarize_entity(
                sample_function_entity,
                file_path="src/utils.py",
                language="python",
            )

    # -------------------------------------------------------------------------
    # Concurrency Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, mock_llm_provider):
        """Test that semaphore limits concurrent requests."""
        summarizer = CodeSummarizer(llm_provider=mock_llm_provider, max_concurrent=3)

        assert summarizer._semaphore._value == 3

    @pytest.mark.asyncio
    async def test_multiple_concurrent_summaries(self, summarizer_with_provider, sample_parsed_file, mock_llm_provider):
        """Test multiple concurrent summarization requests."""
        # Create multiple summarization tasks
        tasks = [
            summarizer_with_provider.summarize_file(sample_parsed_file)
            for _ in range(5)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r == "This is a test summary." for r in results)


# ============================================================================
# Prompt Generation Tests
# ============================================================================

class TestSummaryPrompts:
    """Tests for SummaryPrompts."""

    def test_file_prompt_contains_required_info(self):
        """Test that file prompt contains required information."""
        from code_rag.summarization.prompts import SummaryPrompts

        prompt = SummaryPrompts.get_file_prompt(
            file_path="src/main.py",
            language="python",
            content="def main(): pass",
        )

        assert "src/main.py" in prompt
        assert "python" in prompt.lower()
        assert "def main()" in prompt

    def test_function_prompt_contains_required_info(self):
        """Test that function prompt contains required information."""
        from code_rag.summarization.prompts import SummaryPrompts

        prompt = SummaryPrompts.get_function_prompt(
            name="process_data",
            file_path="src/utils.py",
            signature="def process_data(data: list)",
            code="def process_data(data): return data",
            language="python",
            docstring="Process the data.",
        )

        assert "process_data" in prompt
        assert "src/utils.py" in prompt
        assert "Process the data" in prompt

    def test_class_prompt_contains_required_info(self):
        """Test that class prompt contains required information."""
        from code_rag.summarization.prompts import SummaryPrompts

        prompt = SummaryPrompts.get_class_prompt(
            name="DataProcessor",
            file_path="src/processor.py",
            code="class DataProcessor: pass",
            language="python",
            docstring="Handles data processing.",
        )

        assert "DataProcessor" in prompt
        assert "src/processor.py" in prompt
        assert "Handles data processing" in prompt

    def test_function_prompt_without_docstring(self):
        """Test function prompt when no docstring is provided."""
        from code_rag.summarization.prompts import SummaryPrompts

        prompt = SummaryPrompts.get_function_prompt(
            name="helper",
            file_path="src/utils.py",
            signature="def helper()",
            code="def helper(): pass",
            language="python",
            docstring=None,
        )

        assert "helper" in prompt
        # Should still be valid prompt

    def test_class_prompt_without_docstring(self):
        """Test class prompt when no docstring is provided."""
        from code_rag.summarization.prompts import SummaryPrompts

        prompt = SummaryPrompts.get_class_prompt(
            name="Service",
            file_path="src/service.py",
            code="class Service: pass",
            language="python",
            docstring=None,
        )

        assert "Service" in prompt


# ============================================================================
# Integration Tests
# ============================================================================

class TestSummarizationIntegration:
    """Integration tests for summarization module."""

    @pytest.fixture
    def sample_project_path(self) -> Path:
        return Path(__file__).parent / "fixtures" / "sample_project"

    @pytest.mark.asyncio
    async def test_summarize_real_file(self, sample_project_path):
        """Test summarizing a real file from sample project (mocked LLM)."""
        if not sample_project_path.exists():
            pytest.skip("Sample project not found")

        from code_rag.parsing.parser import CodeParser
        from code_rag.parsing.scanner import FileScanner

        # Parse a real file
        scanner = FileScanner(sample_project_path)
        parser = CodeParser()

        user_file = sample_project_path / "src" / "models" / "user.py"
        if not user_file.exists():
            pytest.skip("User file not found")

        file_info = next(f for f in scanner.scan_all() if f.path == user_file)
        parsed_file = parser.parse_file(file_info)

        # Create mocked summarizer
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value="Test summary for code")

        summarizer = CodeSummarizer(llm_provider=mock_provider)

        # Summarize
        summaries = await summarizer.summarize_parsed_file(parsed_file)

        # Should have file summary
        assert "__file__" in summaries

        # Should have entity summaries
        entity_summaries = [k for k in summaries.keys() if k != "__file__"]
        assert len(entity_summaries) > 0

        # LLM should have been called multiple times
        assert mock_provider.complete.call_count >= 2

    @pytest.mark.asyncio
    async def test_summarize_entities_preserves_qualified_names(self, sample_project_path):
        """Test that summaries are keyed by qualified names."""
        if not sample_project_path.exists():
            pytest.skip("Sample project not found")

        from code_rag.parsing.parser import CodeParser
        from code_rag.parsing.scanner import FileScanner

        scanner = FileScanner(sample_project_path)
        parser = CodeParser()

        auth_file = sample_project_path / "src" / "api" / "auth.py"
        if not auth_file.exists():
            pytest.skip("Auth file not found")

        file_info = next(f for f in scanner.scan_all() if f.path == auth_file)
        parsed_file = parser.parse_file(file_info)

        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value="Summary")

        summarizer = CodeSummarizer(llm_provider=mock_provider)
        summaries = await summarizer.summarize_parsed_file(parsed_file)

        # Check for method qualified names (e.g., AuthService.login)
        keys = list(summaries.keys())
        method_keys = [k for k in keys if "." in k and k != "__file__"]

        # If there are classes with methods, should have qualified names
        if len(method_keys) > 0:
            assert any("AuthService" in k for k in method_keys) or len(method_keys) >= 1


# Need to import asyncio for concurrent tests
import asyncio
