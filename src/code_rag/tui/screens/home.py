"""Home screen for Code RAG TUI."""

import logging
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, Label, Static
from textual.worker import Worker, WorkerState

from code_rag.tui.constants import Colors, ScreenNames, WidgetIds
from code_rag.tui.screens.base import BaseScreen

logger = logging.getLogger(__name__)


class HomeScreen(BaseScreen):
    """Home screen with repository path input."""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+s", "settings", "Settings"),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repo_path: str = ""
        self._status_worker: Worker | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Vertical(
                Static("Code RAG", id="welcome-title"),
                Static(
                    "Index and query your codebase with AI-powered search",
                    classes="subtitle",
                ),
                Static(""),
                Label("Repository Path:"),
                Input(
                    placeholder="Enter path to your codebase (e.g., /path/to/project)",
                    id="path-input",
                ),
                Static("", id="index-status", classes="status-text"),
                Horizontal(
                    Button("Index", id="index-btn", variant="primary"),
                    Button("Query", id="query-btn", variant="success", disabled=True),
                    id="action-buttons",
                ),
                Horizontal(
                    Button("Manage Projects", id="projects-btn", variant="default"),
                    Button("Settings", id="settings-btn"),
                    id="secondary-buttons",
                ),
                id="welcome-box",
            ),
            id="home-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(WidgetIds.PATH_INPUT, Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "path-input":
            self.repo_path = event.value
            self._check_index_status(event.value)

    def _check_index_status(self, path: str) -> None:
        status = self.query_one(WidgetIds.INDEX_STATUS, Static)
        query_btn = self.query_one(WidgetIds.QUERY_BTN, Button)

        is_valid, error_msg, path_obj = self._validate_path(path)

        if not path:
            status.update("")
            query_btn.disabled = True
            return

        if not is_valid:
            status.update(f"{Colors.ERROR}{error_msg}[/]")
            query_btn.disabled = True
            return

        self._status_worker = self.run_worker(
            self._check_index_async(str(path_obj)),
            name="status_check",
            exclusive=True,
        )

    def _validate_path(self, path: str) -> tuple[bool, str, Path | None]:
        """Validate path and return (is_valid, error_message, resolved_path)."""
        path = path.strip()
        if not path:
            return False, "", None

        path_obj = self._resolve_path(path)

        if not path_obj.exists():
            return False, "Path does not exist or is not a directory", None

        if not path_obj.is_dir():
            return False, "Path does not exist or is not a directory", None

        return True, "", path_obj

    def _resolve_path(self, path: str) -> Path:
        """Resolve path to absolute Path object."""
        return Path(path).expanduser().resolve()

    async def _check_index_async(self, path: str) -> dict:
        """Check index status asynchronously."""
        from code_rag.graph.client import MemgraphClient

        result = {"indexed": False, "files": 0, "nodes": 0}
        try:
            async with MemgraphClient() as client:
                query = """
                MATCH (f:File)
                WHERE f.path STARTS WITH $path
                RETURN count(f) as file_count
                """
                res = await client.execute(query, {"path": path})
                if res and res[0].get("file_count", 0) > 0:
                    result["indexed"] = True
                    result["files"] = res[0]["file_count"]

                    query2 = """
                    MATCH (n)
                    WHERE (n:File OR n:Class OR n:Function OR n:Method)
                    AND (n.file_path STARTS WITH $path OR n.path STARTS WITH $path)
                    RETURN count(n) as node_count
                    """
                    res2 = await client.execute(query2, {"path": path})
                    if res2:
                        result["nodes"] = res2[0].get("node_count", 0)
        except Exception as e:
            logger.error(f"Error checking index status: {e}")

        return result

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name == "status_check" and event.state == WorkerState.SUCCESS:
            result = event.worker.result
            status = self.query_one(WidgetIds.INDEX_STATUS, Static)
            query_btn = self.query_one(WidgetIds.QUERY_BTN, Button)

            if result.get("indexed"):
                status.update(
                    f"{Colors.SUCCESS}Already indexed:[/] {result['files']} files, "
                    f"{result['nodes']} entities. Click Query to search or Index to refresh."
                )
                query_btn.disabled = False
            else:
                status.update(f"{Colors.WARNING}Not indexed yet.[/] Click Index to start.")
                query_btn.disabled = True

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "path-input":
            self._start_indexing()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "index-btn":
            self._start_indexing()
        elif event.button.id == "query-btn":
            self._start_query()
        elif event.button.id == "projects-btn":
            self.app.push_screen(ScreenNames.PROJECTS)
        elif event.button.id == "settings-btn":
            self.app.push_screen(ScreenNames.SETTINGS)

    def _start_indexing(self) -> None:
        path = self.query_one(WidgetIds.PATH_INPUT, Input).value.strip()

        is_valid, error_msg, path_obj = self._validate_path(path)

        if not path:
            self.notify("Please enter a repository path", severity="error")
            return

        if not is_valid:
            self.notify(error_msg, severity="error")
            return

        logger.info(f"Starting indexing for path: {path_obj}")
        self.app.repo_path = str(path_obj)
        self.app.push_screen(ScreenNames.INDEXING)

    def _start_query(self) -> None:
        path = self.query_one(WidgetIds.PATH_INPUT, Input).value.strip()

        if path:
            is_valid, _, path_obj = self._validate_path(path)
            if is_valid:
                self.app.repo_path = str(path_obj)
                logger.info(f"Opening query screen for path: {path_obj}")

        self.app.push_screen(ScreenNames.QUERY)

    def action_settings(self) -> None:
        self.app.push_screen(ScreenNames.SETTINGS)

    def action_quit(self) -> None:
        self.app.exit()
