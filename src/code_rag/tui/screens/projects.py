"""Projects management screen for Code RAG TUI."""

import logging

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, DataTable, Footer, Header, Label, Static
from textual.worker import Worker, WorkerState

from code_rag.projects.manager import ProjectManager
from code_rag.projects.models import Project
from code_rag.tui.constants import Colors, WidgetIds
from code_rag.tui.screens.base import BaseScreen

logger = logging.getLogger(__name__)


class ProjectsScreen(BaseScreen):
    """Screen for managing projects."""

    class ProjectSelected(Message):
        """Message when a project is selected."""

        def __init__(self, project_name: str) -> None:
            self.project_name = project_name
            super().__init__()

    class ProjectDeleted(Message):
        """Message when a project is deleted."""

        def __init__(self, project_name: str) -> None:
            self.project_name = project_name
            super().__init__()

    BINDINGS = [
        ("escape", "go_home", "Home"),
        ("r", "refresh", "Refresh"),
        ("d", "delete", "Delete"),
        ("enter", "view_details", "View Details"),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._projects: list[Project] = []
        self._selected_project: str | None = None
        self._load_worker: Worker | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Vertical(
                Static("Manage Projects", id="projects-title"),
                Static(
                    "View, manage, and delete indexed projects",
                    classes="subtitle",
                ),
                Static(""),
                Container(
                    DataTable(id="projects-table"),
                    id="table-container",
                ),
                Label("", id="status-label"),
                Horizontal(
                    Button("Refresh", id="refresh-btn", variant="primary"),
                    Button("View Details", id="details-btn", variant="success"),
                    Button("Delete", id="delete-btn", variant="error"),
                    Button("Back", id="back-btn"),
                    id="projects-buttons",
                ),
                id="projects-box",
            ),
            id="projects-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(WidgetIds.PROJECTS_TABLE, DataTable)
        table.add_columns("Project", "Files", "Entities", "Chunks", "Last Indexed")
        table.cursor_type = "row"
        self._load_projects()

    def _load_projects(self) -> None:
        self.update_status("Loading projects...")
        logger.info("Loading projects from database")
        self._load_worker = self.run_worker(
            self._fetch_projects(),
            name="load_projects",
        )

    async def _fetch_projects(self) -> list[Project]:
        async with ProjectManager() as manager:
            return await manager.list_projects()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name == "load_projects":
            if event.state == WorkerState.SUCCESS:
                self._projects = event.worker.result
                self._update_table()
                logger.info(f"Loaded {len(self._projects)} projects")

                if self._projects:
                    self.update_status(f"Found {len(self._projects)} project(s)")
                else:
                    self.update_status("No projects found. Index a repository first.")
            elif event.state == WorkerState.ERROR:
                logger.error(f"Error loading projects: {event.worker.error}")
                self.update_status(f"Error loading projects: {event.worker.error}", Colors.ERROR)

        elif event.worker.name == "delete_project":
            if event.state == WorkerState.SUCCESS:
                logger.info(f"Project '{self._selected_project}' deleted")
                self.notify(f"Project '{self._selected_project}' deleted", severity="information")
                self.post_message(self.ProjectDeleted(self._selected_project))
                self._load_projects()
            elif event.state == WorkerState.ERROR:
                logger.error(f"Error deleting project: {event.worker.error}")
                self.notify(f"Error deleting project: {event.worker.error}", severity="error")

    def _update_table(self) -> None:
        table = self.query_one(WidgetIds.PROJECTS_TABLE, DataTable)
        table.clear()

        for project in self._projects:
            last_indexed = ""
            if project.last_indexed_at:
                last_indexed = project.last_indexed_at.strftime("%Y-%m-%d %H:%M")

            table.add_row(
                project.name,
                str(project.total_files),
                str(project.total_entities),
                str(project.total_chunks),
                last_indexed,
                key=project.name,
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._selected_project = str(event.row_key.value)
        self.update_status(f"Selected: {self._selected_project}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "refresh-btn":
            self._load_projects()
        elif event.button.id == "details-btn":
            self._view_details()
        elif event.button.id == "delete-btn":
            self._delete_project()
        elif event.button.id == "back-btn":
            self.app.pop_screen()

    def _view_details(self) -> None:
        if not self._selected_project:
            self.notify("Select a project first", severity="warning")
            return

        project = next(
            (p for p in self._projects if p.name == self._selected_project),
            None,
        )
        if project:
            from code_rag.tui.screens.project_detail import ProjectDetailScreen

            logger.info(f"Opening details for project: {self._selected_project}")
            self.app.push_screen(ProjectDetailScreen(project))

    def _delete_project(self) -> None:
        if not self._selected_project:
            self.notify("Select a project first", severity="warning")
            return

        from code_rag.tui.screens.confirm_delete import ConfirmDeleteScreen

        logger.info(f"Requesting deletion confirmation for project: {self._selected_project}")
        self.app.push_screen(
            ConfirmDeleteScreen(self._selected_project),
            self._on_delete_confirmed,
        )

    def _on_delete_confirmed(self, confirmed: bool) -> None:
        if confirmed and self._selected_project:
            self.update_status(f"Deleting {self._selected_project}...")
            logger.info(f"Deleting project: {self._selected_project}")
            self.run_worker(
                self._do_delete(self._selected_project),
                name="delete_project",
            )

    async def _do_delete(self, project_name: str) -> bool:
        async with ProjectManager() as manager:
            return await manager.delete_project(project_name)

    def action_refresh(self) -> None:
        self._load_projects()

    def action_delete(self) -> None:
        self._delete_project()

    def action_view_details(self) -> None:
        self._view_details()

    def action_go_home(self) -> None:
        self.app.pop_screen()
