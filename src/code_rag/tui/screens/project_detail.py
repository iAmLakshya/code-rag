"""Project detail screen for Code RAG TUI."""

import logging
from datetime import datetime

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, DataTable, Footer, Header, Label, Static
from textual.worker import Worker, WorkerState

from code_rag.projects.manager import ProjectManager
from code_rag.projects.models import Project
from code_rag.tui.screens.base import BaseScreen

logger = logging.getLogger(__name__)


class ProjectDetailScreen(BaseScreen):
    """Screen showing project details."""

    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("d", "delete_index", "Delete Index"),
    ]

    def __init__(self, project: Project, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project = project

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Vertical(
                Static(f"Project: {self.project.name}", id="detail-title"),
                Static(""),
                Container(
                    Static(f"[bold]Created:[/bold] {self._format_datetime(self.project.created_at)}"),
                    Static(
                        f"[bold]Last Indexed:[/bold] {self._format_datetime(self.project.last_indexed_at)}"
                    ),
                    Static(f"[bold]Total Files:[/bold] {self.project.total_files}"),
                    Static(f"[bold]Total Entities:[/bold] {self.project.total_entities}"),
                    Static(f"[bold]Total Chunks:[/bold] {self.project.total_chunks}"),
                    id="project-stats",
                ),
                Static(""),
                Static("[bold]Indexed Paths:[/bold]", id="indexes-title"),
                Container(
                    DataTable(id="indexes-table"),
                    id="indexes-container",
                ),
                Label("", id="detail-status"),
                Horizontal(
                    Button("Delete Selected Index", id="delete-index-btn", variant="error"),
                    Button("Back", id="back-btn"),
                    id="detail-buttons",
                ),
                id="detail-box",
            ),
            id="detail-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#indexes-table", DataTable)
        table.add_columns("Path", "Files", "Entities", "Chunks", "Indexed At")
        table.cursor_type = "row"

        for idx in self.project.indexes:
            indexed_at = self._format_datetime(idx.indexed_at)
            table.add_row(
                idx.path,
                str(idx.file_count),
                str(idx.entity_count),
                str(idx.chunk_count),
                indexed_at,
                key=idx.path,
            )

    def _format_datetime(self, dt: datetime | None) -> str:
        """Format datetime for display."""
        if dt is None:
            return "N/A"
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "delete-index-btn":
            self._delete_index()
        elif event.button.id == "back-btn":
            self.app.pop_screen()

    def _delete_index(self) -> None:
        table = self.query_one("#indexes-table", DataTable)
        if table.cursor_row is None:
            self.notify("Select an index first", severity="warning")
            return

        row_key = table.get_row_at(table.cursor_row)
        if row_key:
            path = str(table.get_cell_at((table.cursor_row, 0)))
            logger.info(f"Deleting index for path: {path}")
            self.run_worker(
                self._do_delete_index(path),
                name="delete_index",
            )

    async def _do_delete_index(self, path: str) -> bool:
        async with ProjectManager() as manager:
            return await manager.delete_index(self.project.name, path)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name == "delete_index":
            if event.state == WorkerState.SUCCESS:
                logger.info("Index deleted successfully")
                self.notify("Index deleted", severity="information")
                self.app.pop_screen()
            elif event.state == WorkerState.ERROR:
                logger.error(f"Error deleting index: {event.worker.error}")
                self.notify(f"Error: {event.worker.error}", severity="error")

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_delete_index(self) -> None:
        self._delete_index()
