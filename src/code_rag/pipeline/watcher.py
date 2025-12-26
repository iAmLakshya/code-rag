"""Real-time file watching and graph updates.

Monitors a repository for file changes and incrementally updates the
knowledge graph without requiring a full re-index.

Usage:
    # Start watching a repository
    watcher = FileWatcher(repo_path, graph_builder, vector_indexer)
    await watcher.start()

    # Stop watching
    await watcher.stop()
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from code_rag.config import get_settings
from code_rag.core.cache import ASTCache
from code_rag.parsing.language_config import get_config_for_file

if TYPE_CHECKING:
    from code_rag.embeddings.indexer import VectorIndexer
    from code_rag.graph.builder import GraphBuilder
    from code_rag.parsing.parser import CodeParser

logger = logging.getLogger(__name__)


def _import_watchdog():
    """Import watchdog with helpful error message."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        return FileSystemEventHandler, Observer
    except ImportError:
        raise ImportError(
            "Real-time updates require the 'watchdog' package. "
            "Install with: pip install watchdog"
        )


class FileChangeHandler:
    """Handles file system events and triggers graph updates.

    This class processes file changes (create, modify, delete) and
    coordinates updates to both the graph database and vector store.
    """

    def __init__(
        self,
        repo_path: Path,
        on_file_changed: Callable[[Path], None],
        on_file_deleted: Callable[[Path], None],
    ):
        """Initialize the file change handler.

        Args:
            repo_path: Root path of the repository.
            on_file_changed: Callback for file changes.
            on_file_deleted: Callback for file deletions.
        """
        self.repo_path = repo_path
        self.on_file_changed = on_file_changed
        self.on_file_deleted = on_file_deleted

        settings = get_settings()
        self.ignore_patterns = set(settings.ignore_patterns)
        self.supported_extensions = set(settings.supported_extensions)

        # Debounce: track pending changes
        self._pending_changes: dict[Path, str] = {}
        self._debounce_delay = 0.5  # seconds

    def _is_relevant(self, path: Path) -> bool:
        """Check if a file path should trigger updates.

        Args:
            path: File path to check.

        Returns:
            True if file is relevant for indexing.
        """
        # Check extension
        if path.suffix.lower() not in self.supported_extensions:
            return False

        # Check ignore patterns
        try:
            relative = path.relative_to(self.repo_path)
            for part in relative.parts:
                if part in self.ignore_patterns:
                    return False
        except ValueError:
            return False

        return True

    def handle_event(self, event_type: str, src_path: str) -> None:
        """Handle a file system event.

        Args:
            event_type: Type of event (created, modified, deleted).
            src_path: Path to the affected file.
        """
        path = Path(src_path)

        # Skip directories and irrelevant files
        if path.is_dir() or not self._is_relevant(path):
            return

        logger.info(f"File {event_type}: {path}")

        if event_type in ("created", "modified"):
            self.on_file_changed(path)
        elif event_type == "deleted":
            self.on_file_deleted(path)


class FileWatcher:
    """Watches a repository for changes and updates the graph incrementally.

    Features:
    - Watches for file creates, modifies, and deletes
    - Filters by supported extensions and ignore patterns
    - Debounces rapid changes to avoid excessive updates
    - Coordinates graph and vector store updates
    """

    def __init__(
        self,
        repo_path: Path,
        graph_builder: "GraphBuilder",
        vector_indexer: "VectorIndexer",
        parser: "CodeParser",
        ast_cache: ASTCache | None = None,
    ):
        """Initialize the file watcher.

        Args:
            repo_path: Root path of the repository to watch.
            graph_builder: Graph builder for graph updates.
            vector_indexer: Vector indexer for embedding updates.
            parser: Code parser for parsing changed files.
            ast_cache: Optional AST cache for performance.
        """
        self.repo_path = repo_path.resolve()
        self.graph_builder = graph_builder
        self.vector_indexer = vector_indexer
        self.parser = parser
        self.ast_cache = ast_cache or ASTCache()

        self._observer = None
        self._running = False
        self._update_queue: asyncio.Queue[tuple[str, Path]] = asyncio.Queue()
        self._update_task = None

        # Statistics
        self.files_updated = 0
        self.files_deleted = 0
        self.errors = 0

    async def start(self) -> None:
        """Start watching for file changes.

        This is a non-blocking call. Use `run_forever()` to block.
        """
        if self._running:
            logger.warning("Watcher already running")
            return

        FileSystemEventHandler, Observer = _import_watchdog()

        # Create watchdog handler
        class WatchdogHandler(FileSystemEventHandler):
            def __init__(handler_self, change_handler: FileChangeHandler):
                super().__init__()
                handler_self.change_handler = change_handler

            def on_created(handler_self, event):
                if not event.is_directory:
                    handler_self.change_handler.handle_event("created", event.src_path)

            def on_modified(handler_self, event):
                if not event.is_directory:
                    handler_self.change_handler.handle_event("modified", event.src_path)

            def on_deleted(handler_self, event):
                if not event.is_directory:
                    handler_self.change_handler.handle_event("deleted", event.src_path)

        # Create change handler
        change_handler = FileChangeHandler(
            repo_path=self.repo_path,
            on_file_changed=lambda p: self._queue_update("changed", p),
            on_file_deleted=lambda p: self._queue_update("deleted", p),
        )

        # Start observer
        self._observer = Observer()
        handler = WatchdogHandler(change_handler)
        self._observer.schedule(handler, str(self.repo_path), recursive=True)
        self._observer.start()
        self._running = True

        # Start update processor
        self._update_task = asyncio.create_task(self._process_updates())

        logger.info(f"Started watching: {self.repo_path}")

    async def stop(self) -> None:
        """Stop watching for file changes."""
        if not self._running:
            return

        self._running = False

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None

        logger.info(f"Stopped watching: {self.repo_path}")
        logger.info(f"Stats: {self.files_updated} updated, {self.files_deleted} deleted, {self.errors} errors")

    async def run_forever(self) -> None:
        """Run the watcher until interrupted.

        This is a blocking call.
        """
        await self.start()
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    def _queue_update(self, action: str, path: Path) -> None:
        """Queue a file update for processing.

        Args:
            action: "changed" or "deleted".
            path: File path.
        """
        try:
            self._update_queue.put_nowait((action, path))
        except asyncio.QueueFull:
            logger.warning(f"Update queue full, dropping: {path}")

    async def _process_updates(self) -> None:
        """Process queued file updates."""
        while self._running:
            try:
                # Wait for an update
                action, path = await asyncio.wait_for(
                    self._update_queue.get(),
                    timeout=1.0,
                )

                # Process the update
                if action == "changed":
                    await self._handle_file_changed(path)
                elif action == "deleted":
                    await self._handle_file_deleted(path)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing update: {e}")
                self.errors += 1

    async def _handle_file_changed(self, file_path: Path) -> None:
        """Handle a file create or modify event.

        Args:
            file_path: Path to the changed file.
        """
        logger.info(f"Processing file change: {file_path}")

        try:
            # Check if file still exists
            if not file_path.exists():
                logger.debug(f"File no longer exists: {file_path}")
                return

            # Get language config
            lang_config = get_config_for_file(file_path)
            if not lang_config:
                logger.debug(f"Unsupported file type: {file_path}")
                return

            # Parse the file
            from code_rag.parsing.models import FileInfo
            from code_rag.core.types import Language

            # Create FileInfo
            content = file_path.read_text(encoding="utf-8", errors="replace")
            import hashlib

            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Map language name to enum
            lang_map = {
                "python": Language.PYTHON,
                "javascript": Language.JAVASCRIPT,
                "typescript": Language.TYPESCRIPT,
                "jsx": Language.JSX,
                "tsx": Language.TSX,
            }
            language = lang_map.get(lang_config.name, Language.PYTHON)

            file_info = FileInfo(
                path=file_path,
                relative_path=str(file_path.relative_to(self.repo_path)),
                language=language,
                content_hash=content_hash,
                size_bytes=len(content.encode()),
                line_count=content.count("\n") + 1,
            )

            # Parse
            parsed_file = await asyncio.to_thread(
                self.parser.parse_file, file_info
            )

            if parsed_file:
                # Update graph
                relative_path = str(file_path.relative_to(self.repo_path))

                # Delete old entities
                await self.graph_builder.delete_file_entities(relative_path)

                # Build new entities
                await self.graph_builder.build_from_parsed_file(parsed_file)

                # Update embeddings
                await self.vector_indexer.index_file(
                    parsed_file,
                    project_name=self.repo_path.name,
                )

                # Update AST cache
                if hasattr(parsed_file, "_tree") and parsed_file._tree:
                    self.ast_cache[file_path] = (
                        parsed_file._tree.root_node,
                        lang_config.name,
                    )

                self.files_updated += 1
                logger.info(f"Updated: {file_path}")

        except Exception as e:
            logger.error(f"Failed to update {file_path}: {e}")
            self.errors += 1

    async def _handle_file_deleted(self, file_path: Path) -> None:
        """Handle a file delete event.

        Args:
            file_path: Path to the deleted file.
        """
        logger.info(f"Processing file deletion: {file_path}")

        try:
            relative_path = str(file_path.relative_to(self.repo_path))

            # Delete from graph
            await self.graph_builder.delete_file_entities(relative_path)

            # Delete from vector store
            await self.vector_indexer.delete_file(relative_path)

            # Remove from AST cache
            if file_path in self.ast_cache:
                del self.ast_cache[file_path]

            self.files_deleted += 1
            logger.info(f"Deleted: {file_path}")

        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")
            self.errors += 1


async def start_watcher(
    repo_path: str | Path,
    graph_builder: "GraphBuilder",
    vector_indexer: "VectorIndexer",
    parser: "CodeParser",
) -> FileWatcher:
    """Convenience function to start a file watcher.

    Args:
        repo_path: Repository path to watch.
        graph_builder: Graph builder instance.
        vector_indexer: Vector indexer instance.
        parser: Code parser instance.

    Returns:
        Running FileWatcher instance.
    """
    watcher = FileWatcher(
        repo_path=Path(repo_path),
        graph_builder=graph_builder,
        vector_indexer=vector_indexer,
        parser=parser,
    )
    await watcher.start()
    return watcher
