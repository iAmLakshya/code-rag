"""Constants for TUI module."""


class Colors:
    ERROR = "[red]"
    SUCCESS = "[green]"
    INFO = "[blue]"
    WARNING = "[yellow]"
    DIM = "[dim]"
    BOLD = "[bold]"
    BOLD_BLUE = "[bold blue]"
    BOLD_GREEN = "[bold green]"


class WidgetIds:
    STATUS_LABEL = "#status-label"
    QUERY_BTN = "#query-btn"
    CANCEL_BTN = "#cancel-btn"
    INDEX_BTN = "#index-btn"
    PATH_INPUT = "#path-input"
    INDEX_STATUS = "#index-status"
    QUERY_INPUT = "#query-input"
    CHAT_LOG = "#chat-log"
    SOURCE_PREVIEW = "#source-preview"
    INDEXING_LOG = "#indexing-log"
    REPO_INFO = "#repo-info"
    STATUS_INFO = "#status-info"
    PROJECTS_TABLE = "#projects-table"


class ScreenNames:
    HOME = "home"
    INDEXING = "indexing"
    QUERY = "query"
    SETTINGS = "settings"
    PROJECTS = "projects"
