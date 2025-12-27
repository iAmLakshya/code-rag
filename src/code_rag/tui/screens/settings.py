"""Settings screen for Code RAG TUI."""

import logging

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Button, Footer, Header, Input, Static

from code_rag.config import get_settings
from code_rag.tui.screens.base import BaseScreen

logger = logging.getLogger(__name__)


class SettingsScreen(BaseScreen):
    """Settings configuration screen."""

    BINDINGS = [
        ("escape", "go_back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        settings = get_settings()

        yield Header()
        yield Container(
            Vertical(
                Static("Settings", id="settings-title"),
                Static(""),
                self._create_setting_field(
                    "OpenAI API Key:",
                    "sk-****" if settings.openai_api_key else "",
                    "api-key-input",
                    password=True,
                    placeholder="sk-...",
                ),
                self._create_setting_field(
                    "Memgraph Host:",
                    settings.memgraph_host,
                    "memgraph-host-input",
                ),
                self._create_setting_field(
                    "Qdrant Host:",
                    settings.qdrant_host,
                    "qdrant-host-input",
                ),
                self._create_setting_field(
                    "LLM Model:",
                    settings.llm_model,
                    "llm-model-input",
                ),
                self._create_setting_field(
                    "Embedding Model:",
                    settings.embedding_model,
                    "embedding-model-input",
                ),
                Static(""),
                Static(
                    "Note: Settings are loaded from environment variables or .env file.",
                    id="settings-note",
                ),
                Static(
                    "Modify your .env file to persist changes.",
                    id="settings-note-2",
                ),
                Static(""),
                Button("Back", id="back-btn"),
                id="settings-form",
            ),
            id="settings-container",
        )
        yield Footer()

    def _create_setting_field(
        self,
        label: str,
        value: str,
        input_id: str,
        password: bool = False,
        placeholder: str = "",
    ) -> Container:
        """Create a setting field container."""
        return Container(
            Static(label, classes="setting-label"),
            Input(
                value=value,
                id=input_id,
                password=password,
                placeholder=placeholder,
            ),
            classes="setting-group",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()

    def action_go_back(self) -> None:
        self.app.pop_screen()
