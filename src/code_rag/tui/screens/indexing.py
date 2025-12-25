"""Indexing screen for Code RAG TUI."""

import logging

from textual.app import ComposeResult
from textual.containers import Container, Grid, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Footer, Header, Label, ProgressBar, RichLog, Static
from textual.worker import Worker, WorkerState

from code_rag.core.types import PipelineStage
from code_rag.pipeline.orchestrator import PipelineOrchestrator
from code_rag.pipeline.progress import PipelineProgress
from code_rag.tui.constants import Colors, ScreenNames, WidgetIds
from code_rag.tui.screens.base import BaseScreen

logger = logging.getLogger(__name__)


class IndexingScreen(BaseScreen):
    """Screen showing indexing progress."""

    class ProgressUpdate(Message):
        """Message for progress updates from worker thread."""

        def __init__(self, progress: PipelineProgress) -> None:
            self.progress = progress
            super().__init__()

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("q", "go_home", "Home"),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._worker: Worker | None = None
        self._indexing_complete: bool = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Vertical(
                Static("Indexing Repository", id="indexing-title"),
                Label("", id="repo-path-label"),
                Static(""),
                Label("Overall Progress:", id="progress-label"),
                ProgressBar(total=100, id="overall-progress"),
                Label("Stage: Idle", id="stage-label"),
                ProgressBar(total=100, id="stage-progress"),
                Label("", id="current-file"),
                Container(
                    Grid(
                        Static("Files\n0", classes="stat-box", id="stat-files"),
                        Static("Entities\n0", classes="stat-box", id="stat-entities"),
                        Static("Nodes\n0", classes="stat-box", id="stat-nodes"),
                        Static("Chunks\n0", classes="stat-box", id="stat-chunks"),
                        id="stats-grid",
                    ),
                    id="stats-container",
                ),
                Container(
                    RichLog(id="indexing-log", highlight=True, markup=True),
                    id="log-container",
                ),
                Horizontal(
                    Button("Cancel", id="cancel-btn", variant="error"),
                    Button("Query Index", id="query-btn", variant="success", disabled=True),
                    id="button-container",
                ),
            ),
            id="indexing-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        repo_path = getattr(self.app, "repo_path", "")
        self.query_one("#repo-path-label", Label).update(f"Path: {repo_path}")
        self._start_indexing()

    def _start_indexing(self) -> None:
        repo_path = getattr(self.app, "repo_path", "")
        if not repo_path:
            self.log_message("No repository path set", Colors.ERROR)
            return

        logger.info(f"Starting indexing of {repo_path}")
        self.log_message(f"Starting indexing of {repo_path}", Colors.INFO)
        self._worker = self.run_worker(
            self._run_indexing(repo_path),
            name="indexing",
            exclusive=True,
        )

    async def _run_indexing(self, repo_path: str) -> dict:
        orchestrator = PipelineOrchestrator(
            repo_path=repo_path,
            progress_callback=self._on_progress,
        )
        return await orchestrator.run()

    def _on_progress(self, progress: PipelineProgress) -> None:
        self.post_message(self.ProgressUpdate(progress))

    def on_indexing_screen_progress_update(self, message: ProgressUpdate) -> None:
        self._update_ui(message.progress)

    def _update_ui(self, progress: PipelineProgress) -> None:
        overall_bar = self.query_one("#overall-progress", ProgressBar)
        overall_bar.progress = progress.overall_percentage

        stage_label = self.query_one("#stage-label", Label)
        stage_label.update(f"Stage: {progress.current_stage.value.replace('_', ' ').title()}")

        stage_bar = self.query_one("#stage-progress", ProgressBar)
        if progress.current_stage in progress.stages:
            stage_progress = progress.stages[progress.current_stage]
            if stage_progress.total > 0:
                stage_bar.progress = stage_progress.percentage
            stage_label.update(f"Stage: {stage_progress.message or progress.current_stage.value}")

        if progress.current_stage in progress.stages:
            stage_info = progress.stages[progress.current_stage]
            current_file = self.query_one("#current-file", Label)
            current_file.update(stage_info.message)

        self.query_one("#stat-files", Static).update(f"Files\n{progress.files_parsed}")
        self.query_one("#stat-entities", Static).update(f"Entities\n{progress.entities_found}")
        self.query_one("#stat-nodes", Static).update(f"Nodes\n{progress.graph_nodes_created}")
        self.query_one("#stat-chunks", Static).update(f"Chunks\n{progress.chunks_embedded}")

        if progress.current_stage == PipelineStage.COMPLETED:
            self.log_message(f"Indexing complete in {progress.elapsed_time:.1f}s", Colors.SUCCESS)
        elif progress.current_stage == PipelineStage.FAILED:
            self.log_message(f"Error: {progress.error_message}", Colors.ERROR)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "indexing":
            return

        if event.state == WorkerState.SUCCESS:
            self._handle_worker_success(event.worker.result)
        elif event.state == WorkerState.ERROR:
            self._handle_worker_error(event.worker.error)
        elif event.state == WorkerState.CANCELLED:
            self._handle_worker_cancelled()

    def _handle_worker_success(self, result: dict) -> None:
        self._indexing_complete = True
        logger.info(f"Indexing completed successfully: {result}")

        self.log_message("Indexing completed successfully!", Colors.SUCCESS)
        self.log_message(f"  Files indexed: {result.get('files_indexed', 0)}")
        self.log_message(f"  Entities found: {result.get('entities_found', 0)}")
        self.log_message(f"  Graph nodes: {result.get('graph_nodes', 0)}")
        self.log_message(f"  Chunks embedded: {result.get('chunks_embedded', 0)}")
        self.log_message(f"  Time elapsed: {result.get('elapsed_seconds', 0):.1f}s")
        self.log_message("")
        self.log_message("Click 'Query Index' to start searching your codebase!", Colors.BOLD)

        cancel_btn = self.query_one(WidgetIds.CANCEL_BTN, Button)
        cancel_btn.label = "Close"
        cancel_btn.variant = "default"

        query_btn = self.query_one(WidgetIds.QUERY_BTN, Button)
        query_btn.disabled = False

    def _handle_worker_error(self, error: Exception) -> None:
        logger.error(f"Indexing failed: {error}")

        self.log_message("Indexing failed!", Colors.ERROR)
        self.log_message(f"Error: {error}", Colors.ERROR)

        error_str = str(error).lower()
        if "connection" in error_str or "connect" in error_str:
            self.log_message("Hint: Make sure Docker containers are running:", Colors.WARNING)
            self.log_message("  docker-compose up -d", Colors.WARNING)
        elif "openai" in error_str or "api" in error_str:
            self.log_message("Hint: Check your OPENAI_API_KEY in .env file", Colors.WARNING)

        btn = self.query_one(WidgetIds.CANCEL_BTN, Button)
        btn.label = "Close"
        btn.variant = "error"

    def _handle_worker_cancelled(self) -> None:
        logger.info("Indexing cancelled by user")
        self.log_message("Indexing cancelled", Colors.WARNING)

        btn = self.query_one(WidgetIds.CANCEL_BTN, Button)
        btn.label = "Close"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            if self._worker and self._worker.is_running:
                self._worker.cancel()
            else:
                self.app.pop_screen()
        elif event.button.id == "query-btn":
            self.app.pop_screen()
            self.app.push_screen(ScreenNames.QUERY)

    def action_cancel(self) -> None:
        if self._worker and self._worker.is_running:
            self._worker.cancel()
        else:
            self.app.pop_screen()

    def action_go_home(self) -> None:
        if self._worker and self._worker.is_running:
            self._worker.cancel()
        self.app.pop_screen()
