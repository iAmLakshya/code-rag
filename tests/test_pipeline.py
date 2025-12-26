"""Tests for the indexing pipeline."""

import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from code_rag.pipeline.orchestrator import PipelineOrchestrator
from code_rag.pipeline.progress import ProgressTracker
from code_rag.core.types import PipelineStage
from code_rag.parsing.scanner import FileScanner
from code_rag.parsing.parser import CodeParser


class TestProgressTracker:
    """Tests for ProgressTracker."""

    def test_initial_state(self):
        """Test tracker starts in scanning state."""
        tracker = ProgressTracker()
        assert tracker.progress.current_stage == PipelineStage.SCANNING
        assert not tracker.progress.is_running
        assert not tracker.progress.is_complete

    def test_start(self):
        """Test starting the tracker."""
        tracker = ProgressTracker()
        tracker.start()
        assert tracker.progress.is_running
        assert tracker.progress.start_time is not None

    def test_set_stage(self):
        """Test setting stages."""
        tracker = ProgressTracker()
        tracker.start()
        tracker.set_stage(PipelineStage.PARSING, total=10, message="Parsing...")

        assert tracker.progress.current_stage == PipelineStage.PARSING
        assert PipelineStage.PARSING in tracker.progress.stages
        assert tracker.progress.stages[PipelineStage.PARSING].total == 10

    def test_update_stage(self):
        """Test updating stage progress."""
        tracker = ProgressTracker()
        tracker.start()
        tracker.set_stage(PipelineStage.PARSING, total=10)
        tracker.update_stage(5)

        assert tracker.progress.stages[PipelineStage.PARSING].current == 5
        assert tracker.progress.stages[PipelineStage.PARSING].percentage == 50.0

    def test_complete(self):
        """Test completing the pipeline."""
        tracker = ProgressTracker()
        tracker.start()
        tracker.complete()

        assert tracker.progress.current_stage == PipelineStage.COMPLETED
        assert tracker.progress.is_complete
        assert not tracker.progress.is_running
        assert tracker.progress.end_time is not None

    def test_error(self):
        """Test error state."""
        tracker = ProgressTracker()
        tracker.start()
        tracker.error("Something went wrong")

        assert tracker.progress.current_stage == PipelineStage.FAILED
        assert tracker.progress.has_error
        assert tracker.progress.error_message == "Something went wrong"

    def test_callbacks(self):
        """Test progress callbacks."""
        tracker = ProgressTracker()
        callback_calls = []

        def callback(progress):
            callback_calls.append(progress.current_stage)

        tracker.add_callback(callback)
        tracker.start()
        tracker.set_stage(PipelineStage.PARSING)
        tracker.complete()

        assert len(callback_calls) >= 3
        assert PipelineStage.COMPLETED in callback_calls


class TestPipelineScanAndParse:
    """Tests for scanning and parsing stages (no external dependencies)."""

    @pytest.mark.asyncio
    async def test_scan_files(self, sample_project_path: Path):
        """Test file scanning stage."""
        # Create orchestrator but don't run full pipeline
        orchestrator = PipelineOrchestrator(sample_project_path)

        # Test scanning directly
        scanner = FileScanner(sample_project_path)
        files = scanner.scan_all()

        assert len(files) > 0
        assert orchestrator.repo_path == sample_project_path

    @pytest.mark.asyncio
    async def test_parse_files(self, sample_project_path: Path):
        """Test file parsing stage."""
        scanner = FileScanner(sample_project_path)
        parser = CodeParser()

        files = scanner.scan_all()
        parsed_files = []
        errors = []

        for file_info in files:
            try:
                parsed = parser.parse_file(file_info)
                parsed_files.append(parsed)
            except Exception as e:
                errors.append((file_info.relative_path, str(e)))

        assert len(parsed_files) > 0, "Should parse some files"
        assert len(errors) == 0, f"Should have no parsing errors: {errors}"

        # Check we found entities
        total_entities = sum(len(pf.all_entities) for pf in parsed_files)
        assert total_entities > 0, "Should find entities"


class TestPipelineWithMocks:
    """Tests for pipeline with mocked external services."""

    @pytest.mark.asyncio
    async def test_pipeline_stages_execute(self, sample_project_path: Path):
        """Test that all pipeline stages execute."""
        # Track which stages were reached
        stages_reached = []

        def progress_callback(progress):
            if progress.current_stage not in stages_reached:
                stages_reached.append(progress.current_stage)

        # Create orchestrator
        orchestrator = PipelineOrchestrator(
            sample_project_path,
            progress_callback=progress_callback,
        )

        # Mock the database and API components
        with patch.object(orchestrator, '_init_components', new_callable=AsyncMock) as mock_init, \
             patch.object(orchestrator, '_cleanup', new_callable=AsyncMock) as mock_cleanup, \
             patch.object(orchestrator, '_execute_graph_stage', new_callable=AsyncMock) as mock_graph, \
             patch.object(orchestrator, '_execute_summarize_stage', new_callable=AsyncMock) as mock_summaries, \
             patch.object(orchestrator, '_execute_embedding_stage', new_callable=AsyncMock) as mock_embeddings:

            # Setup mock context
            from code_rag.pipeline.orchestrator import PipelineContext
            mock_ctx = MagicMock(spec=PipelineContext)
            mock_ctx.repo_path = orchestrator.repo_path
            mock_ctx.project_name = orchestrator.project_name
            mock_ctx.tracker = orchestrator.tracker
            mock_ctx.parser = CodeParser()
            mock_ctx.scanned_files = []
            mock_ctx.parsed_files = []
            mock_ctx.file_update_status = {}
            mock_init.return_value = mock_ctx

            try:
                result = await orchestrator.run()
            except Exception as e:
                pytest.fail(f"Pipeline failed: {e}")

        # Check stages were reached
        assert PipelineStage.SCANNING in stages_reached
        assert PipelineStage.PARSING in stages_reached
        assert PipelineStage.COMPLETED in stages_reached

        # Check mocks were called
        mock_init.assert_called_once()
        mock_graph.assert_called_once()


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
class TestPipelineIntegration:
    """Full integration tests (requires API keys and running databases)."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, sample_project_path: Path):
        """Test the full indexing pipeline."""
        progress_updates = []

        def progress_callback(progress):
            progress_updates.append({
                "stage": progress.current_stage.value,
                "files": progress.files_parsed,
                "entities": progress.entities_found,
            })

        orchestrator = PipelineOrchestrator(
            sample_project_path,
            project_name="test_project",
            progress_callback=progress_callback,
        )

        result = await orchestrator.run()

        # Verify results
        assert result["files_indexed"] > 0
        assert result["entities_found"] > 0
        assert result["elapsed_seconds"] > 0

        # Verify progress was tracked
        assert len(progress_updates) > 0

        print(f"\nIndexing results:")
        print(f"  Files: {result['files_indexed']}")
        print(f"  Entities: {result['entities_found']}")
        print(f"  Graph nodes: {result['graph_nodes']}")
        print(f"  Summaries: {result['summaries']}")
        print(f"  Chunks: {result['chunks_embedded']}")
        print(f"  Time: {result['elapsed_seconds']:.2f}s")


class TestPipelineEdgeCases:
    """Tests for edge cases in the pipeline."""

    @pytest.mark.asyncio
    async def test_empty_directory(self, tmp_path: Path):
        """Test pipeline with empty directory."""
        orchestrator = PipelineOrchestrator(tmp_path)

        with patch.object(orchestrator, '_init_components', new_callable=AsyncMock) as mock_init, \
             patch.object(orchestrator, '_cleanup', new_callable=AsyncMock), \
             patch.object(orchestrator, '_execute_graph_stage', new_callable=AsyncMock), \
             patch.object(orchestrator, '_execute_summarize_stage', new_callable=AsyncMock), \
             patch.object(orchestrator, '_execute_embedding_stage', new_callable=AsyncMock):

            from code_rag.pipeline.orchestrator import PipelineContext
            mock_ctx = MagicMock(spec=PipelineContext)
            mock_ctx.repo_path = orchestrator.repo_path
            mock_ctx.project_name = orchestrator.project_name
            mock_ctx.tracker = orchestrator.tracker
            mock_ctx.parser = CodeParser()
            mock_ctx.scanned_files = []
            mock_ctx.parsed_files = []
            mock_ctx.file_update_status = {}
            mock_init.return_value = mock_ctx

            result = await orchestrator.run()

        assert result["files_indexed"] == 0
        assert result["entities_found"] == 0

    @pytest.mark.asyncio
    async def test_directory_with_no_supported_files(self, tmp_path: Path):
        """Test pipeline with unsupported file types."""
        # Create files with unsupported extensions
        (tmp_path / "readme.md").write_text("# README")
        (tmp_path / "data.json").write_text("{}")
        (tmp_path / "config.yaml").write_text("key: value")

        orchestrator = PipelineOrchestrator(tmp_path)

        with patch.object(orchestrator, '_init_components', new_callable=AsyncMock) as mock_init, \
             patch.object(orchestrator, '_cleanup', new_callable=AsyncMock), \
             patch.object(orchestrator, '_execute_graph_stage', new_callable=AsyncMock), \
             patch.object(orchestrator, '_execute_summarize_stage', new_callable=AsyncMock), \
             patch.object(orchestrator, '_execute_embedding_stage', new_callable=AsyncMock):

            from code_rag.pipeline.orchestrator import PipelineContext
            mock_ctx = MagicMock(spec=PipelineContext)
            mock_ctx.repo_path = orchestrator.repo_path
            mock_ctx.project_name = orchestrator.project_name
            mock_ctx.tracker = orchestrator.tracker
            mock_ctx.parser = CodeParser()
            mock_ctx.scanned_files = []
            mock_ctx.parsed_files = []
            mock_ctx.file_update_status = {}
            mock_init.return_value = mock_ctx

            result = await orchestrator.run()

        assert result["files_indexed"] == 0

    def test_nonexistent_directory(self):
        """Test pipeline with non-existent directory."""
        with pytest.raises(ValueError, match="Path does not exist"):
            scanner = FileScanner("/nonexistent/path")
