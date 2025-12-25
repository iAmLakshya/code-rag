"""File scanner for discovering source files in a repository."""

import hashlib
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterator

from code_rag.config import get_settings
from code_rag.core.types import Language
from code_rag.parsing.models import FileInfo


@dataclass
class ScanStatistics:
    file_count: int
    total_lines: int
    total_size: int


class FileScanner:
    """Scans a directory for source files to index."""

    def __init__(
        self,
        root_path: str | Path,
        extensions: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
    ):
        """Initialize the file scanner.

        Args:
            root_path: Root directory to scan.
            extensions: File extensions to include. Defaults to settings.
            ignore_patterns: Directory patterns to ignore. Defaults to settings.
        """
        settings = get_settings()
        self.root_path = Path(root_path).resolve()
        self.extensions = set(extensions or settings.supported_extensions)
        self.ignore_patterns = ignore_patterns or settings.ignore_patterns

        if not self.root_path.exists():
            raise ValueError(f"Path does not exist: {self.root_path}")
        if not self.root_path.is_dir():
            raise ValueError(f"Path is not a directory: {self.root_path}")

    def _should_ignore(self, path: Path) -> bool:
        for pattern in self.ignore_patterns:
            for part in path.parts:
                if fnmatch(part, pattern):
                    return True
        return False

    def _compute_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def scan(self) -> Iterator[FileInfo]:
        """Scan the directory for matching files.

        Yields:
            FileInfo objects for each discovered file.
        """
        for file_path in self.root_path.rglob("*"):
            # Skip directories
            if file_path.is_dir():
                continue

            # Check if should be ignored
            relative = file_path.relative_to(self.root_path)
            if self._should_ignore(relative):
                continue

            # Check extension
            ext = file_path.suffix.lower()
            if ext not in self.extensions:
                continue

            # Get language
            language = Language.from_extension(ext)
            if language is None:
                continue

            # Read file and compute hash
            try:
                content = file_path.read_bytes()
                content_hash = self._compute_hash(content)
                line_count = content.count(b"\n") + 1
            except (OSError, PermissionError):
                continue

            yield FileInfo(
                path=file_path,
                relative_path=str(relative),
                language=language,
                content_hash=content_hash,
                size_bytes=len(content),
                line_count=line_count,
            )

    def scan_all(self) -> list[FileInfo]:
        return list(self.scan())

    def get_statistics(self) -> ScanStatistics:
        file_count = 0
        total_lines = 0
        total_size = 0

        for file_info in self.scan():
            file_count += 1
            total_lines += file_info.line_count
            total_size += file_info.size_bytes

        return ScanStatistics(
            file_count=file_count,
            total_lines=total_lines,
            total_size=total_size,
        )
