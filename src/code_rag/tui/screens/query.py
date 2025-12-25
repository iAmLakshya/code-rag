"""Query screen for Code RAG TUI."""

import logging

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Static
from textual.worker import Worker, WorkerState

from code_rag.query.engine import QueryEngine
from code_rag.tui.constants import Colors, WidgetIds
from code_rag.tui.screens.base import BaseScreen

logger = logging.getLogger(__name__)


class QueryScreen(BaseScreen):
    """Screen for querying the indexed codebase."""

    class EngineReady(Message):
        """Message when engine is ready."""

        pass

    class EngineError(Message):
        """Message when engine fails."""

        def __init__(self, error: str) -> None:
            self.error = error
            super().__init__()

    BINDINGS = [
        ("escape", "go_home", "Home"),
        ("ctrl+l", "clear_chat", "Clear"),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._engine: QueryEngine | None = None
        self._query_worker: Worker | None = None
        self._init_worker: Worker | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Container(
                Label("Repository: Not set", id="repo-info"),
                Label("Status: Connecting...", id="status-info"),
                id="query-header",
            ),
            Vertical(
                Container(
                    Input(
                        placeholder="Ask about your codebase...",
                        id="query-input",
                    ),
                    id="query-input-container",
                ),
                Horizontal(
                    Container(
                        RichLog(id="chat-log", highlight=True, markup=True),
                        id="chat-panel",
                    ),
                    Container(
                        Static("Select a result to view source", id="source-header"),
                        RichLog(id="source-preview", highlight=True),
                        id="source-panel",
                    ),
                    id="results-container",
                ),
                id="query-body",
            ),
            id="query-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        repo_path = getattr(self.app, "repo_path", "Not set")
        self.query_one(WidgetIds.REPO_INFO, Label).update(f"Repository: {repo_path}")

        logger.info("Initializing query engine")
        self._init_worker = self.run_worker(self._initialize_engine(), name="init")

        self.query_one(WidgetIds.QUERY_INPUT, Input).focus()

    async def _initialize_engine(self) -> QueryEngine:
        try:
            engine = QueryEngine()
            await engine.initialize()
            logger.info("Query engine initialized successfully")
            return engine
        except Exception as e:
            logger.error(f"Failed to initialize query engine: {e}")
            raise

    def on_query_screen_engine_ready(self, message: EngineReady) -> None:
        self.query_one(WidgetIds.STATUS_INFO, Label).update("Status: Ready")
        self._log_message("Connected to databases. Ready to query!", "system", "", Colors.DIM)

    def on_query_screen_engine_error(self, message: EngineError) -> None:
        self.query_one(WidgetIds.STATUS_INFO, Label).update("Status: Error")
        self._log_message(f"Failed to connect: {message.error}", "system", "", Colors.ERROR)
        self._log_message(
            "Make sure Docker containers are running:",
            "system",
            "",
            Colors.WARNING,
        )
        self._log_message("  docker-compose up -d", "system", "", "")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "query-input":
            query = event.value.strip()
            if query:
                self._submit_query(query)
                event.input.value = ""

    def _submit_query(self, query: str) -> None:
        if self._engine is None:
            self._log_message("Query engine not ready", "system", "", Colors.ERROR)
            return

        if self._query_worker and self._query_worker.is_running:
            self._log_message("Query in progress...", "system", "", Colors.WARNING)
            return

        logger.info(f"Submitting query: {query}")
        self._log_message(query, "user", "You:", Colors.BOLD_BLUE)

        self._query_worker = self.run_worker(
            self._run_query(query),
            name="query",
        )

    async def _run_query(self, query: str) -> dict:
        result = await self._engine.query(query)
        return {
            "answer": result.answer,
            "sources": [
                {
                    "file_path": s.file_path,
                    "entity_name": s.entity_name,
                    "entity_type": s.entity_type,
                    "content": s.content,
                    "start_line": s.start_line,
                    "end_line": s.end_line,
                    "score": s.score,
                }
                for s in result.sources
            ],
        }

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name == "init":
            if event.state == WorkerState.SUCCESS:
                self._engine = event.worker.result
                self.post_message(self.EngineReady())
            elif event.state == WorkerState.ERROR:
                self.post_message(self.EngineError(str(event.worker.error)))

        elif event.worker.name == "query":
            if event.state == WorkerState.SUCCESS:
                result = event.worker.result
                logger.info(f"Query completed with {len(result['sources'])} sources")
                self._log_message(result["answer"], "assistant", "Assistant:", Colors.BOLD_GREEN)

                if result["sources"]:
                    self._show_source(result["sources"][0])

            elif event.state == WorkerState.ERROR:
                logger.error(f"Query failed: {event.worker.error}")
                self._log_message(f"Query failed: {event.worker.error}", "system", "", Colors.ERROR)

    def _log_message(self, message: str, role: str, prefix: str, style: str) -> None:
        """Unified logging method for all message types."""
        chat = self.query_one(WidgetIds.CHAT_LOG, RichLog)

        if prefix:
            chat.write(f"{style}{prefix}[/]")

        if role == "system":
            chat.write(f"{style}{message}[/]")
        else:
            for line in message.split("\n"):
                chat.write(f"  {line}")

        if role != "system":
            chat.write("")

    def _show_source(self, source: dict) -> None:
        preview = self.query_one(WidgetIds.SOURCE_PREVIEW, RichLog)
        preview.clear()

        header = self.query_one("#source-header", Static)
        header.update(f"{source['file_path']}:{source.get('start_line', '')}")

        if source.get("content"):
            preview.write(f"{Colors.BOLD}{source['entity_name']}[/] ({source['entity_type']})")
            preview.write(f"Lines {source.get('start_line', '?')}-{source.get('end_line', '?')}")
            preview.write("-" * 40)
            preview.write(source["content"])

    def action_go_home(self) -> None:
        self.app.pop_screen()

    def action_clear_chat(self) -> None:
        chat = self.query_one(WidgetIds.CHAT_LOG, RichLog)
        chat.clear()
        logger.info("Chat cleared")

    async def on_unmount(self) -> None:
        if self._engine:
            logger.info("Closing query engine")
            await self._engine.close()
