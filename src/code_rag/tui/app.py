"""Main Textual application for Code RAG."""

import logging

from textual.app import App

from code_rag.tui.constants import ScreenNames
from code_rag.tui.screens.home import HomeScreen
from code_rag.tui.screens.indexing import IndexingScreen
from code_rag.tui.screens.projects import ProjectsScreen
from code_rag.tui.screens.query import QueryScreen
from code_rag.tui.screens.settings import SettingsScreen

logger = logging.getLogger(__name__)


class CodeRAGApp(App):
    """Code RAG Terminal User Interface."""

    TITLE = "Code RAG"
    SUB_TITLE = "AI-powered code search"
    CSS_PATH = "styles.css"

    SCREENS = {
        ScreenNames.HOME: HomeScreen,
        ScreenNames.INDEXING: IndexingScreen,
        ScreenNames.QUERY: QueryScreen,
        ScreenNames.SETTINGS: SettingsScreen,
        ScreenNames.PROJECTS: ProjectsScreen,
    }

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+h", "go_home", "Home"),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repo_path: str = ""

    def on_mount(self) -> None:
        logger.info("Code RAG TUI started")
        self.push_screen(ScreenNames.HOME)

    def action_go_home(self) -> None:
        logger.debug("Navigating to home screen")
        while len(self.screen_stack) > 1:
            self.pop_screen()

    def action_quit(self) -> None:
        logger.info("Code RAG TUI exiting")
        self.exit()


def run_app() -> None:
    """Run the Code RAG TUI application."""
    app = CodeRAGApp()
    app.run()


if __name__ == "__main__":
    run_app()
