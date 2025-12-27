"""Base screen with common utilities."""

import logging
from typing import TypeVar

from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Label, RichLog

T = TypeVar("T", bound=Widget)

logger = logging.getLogger(__name__)


class BaseScreen(Screen):
    """Base screen with common utilities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._widget_cache: dict[str, Widget] = {}

    def get_widget(self, selector: str, widget_type: type[T]) -> T:
        """Get widget with caching."""
        if selector not in self._widget_cache:
            self._widget_cache[selector] = self.query_one(selector, widget_type)
        return self._widget_cache[selector]

    def update_status(self, message: str, style: str = "") -> None:
        """Update status label if exists."""
        try:
            status_widget = self.query_one("#status-label", Label)
            if style:
                status_widget.update(f"{style}{message}[/]")
            else:
                status_widget.update(message)
        except Exception as e:
            logger.debug(f"Failed to update status: {e}")

    def log_message(self, message: str, style: str = "") -> None:
        """Log to RichLog if exists."""
        try:
            log_widget = self.query_one("#indexing-log", RichLog)
            if style:
                log_widget.write(f"{style}{message}[/]")
            else:
                log_widget.write(message)
        except Exception:
            try:
                log_widget = self.query_one("#chat-log", RichLog)
                if style:
                    log_widget.write(f"{style}{message}[/]")
                else:
                    log_widget.write(message)
            except Exception as e:
                logger.debug(f"Failed to log message: {e}")
