"""Generic confirmation screen for deletion."""

import logging

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Static

from code_rag.tui.constants import Colors
from code_rag.tui.screens.base import BaseScreen

logger = logging.getLogger(__name__)


class ConfirmDeleteScreen(BaseScreen):
    """Confirmation screen for deletion."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("y", "confirm", "Yes"),
        ("n", "cancel", "No"),
    ]

    def __init__(
        self,
        item_name: str,
        item_type: str = "project",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.item_name = item_name
        self.item_type = item_type

    def compose(self) -> ComposeResult:
        yield Container(
            Vertical(
                Static("Confirm Deletion", id="confirm-title"),
                Static(""),
                Static(
                    f"Are you sure you want to delete {self.item_type} '{Colors.BOLD}{self.item_name}[/]'?",
                    id="confirm-message",
                ),
                Static(""),
                Static(
                    f"{Colors.WARNING}This will delete all indexed data including:[/]",
                ),
                Static("  - Graph nodes and relationships"),
                Static("  - Vector embeddings"),
                Static("  - Generated summaries"),
                Static(""),
                Static(f"{Colors.ERROR}This action cannot be undone![/]"),
                Static(""),
                Horizontal(
                    Button("Yes, Delete", id="yes-btn", variant="error"),
                    Button("Cancel", id="no-btn", variant="primary"),
                    id="confirm-buttons",
                ),
                id="confirm-box",
            ),
            id="confirm-container",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "yes-btn":
            logger.info(f"User confirmed deletion of {self.item_type}: {self.item_name}")
            self.dismiss(True)
        elif event.button.id == "no-btn":
            logger.info(f"User cancelled deletion of {self.item_type}: {self.item_name}")
            self.dismiss(False)

    def action_confirm(self) -> None:
        logger.info(f"User confirmed deletion of {self.item_type}: {self.item_name}")
        self.dismiss(True)

    def action_cancel(self) -> None:
        logger.info(f"User cancelled deletion of {self.item_type}: {self.item_name}")
        self.dismiss(False)
