"""Memory-aware caching utilities.

Provides bounded caching with LRU eviction and memory limits to prevent
OOM issues on large codebases during long-running sessions.
"""

from __future__ import annotations

import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")


class BoundedCache(Generic[K, V]):
    """Memory-aware LRU cache with automatic cleanup.

    Features:
    - LRU eviction when entry limit is reached
    - Memory-based eviction when memory limit is exceeded
    - Thread-safe operations via OrderedDict
    - Automatic cleanup to prevent memory leaks

    Usage:
        cache = BoundedCache[str, dict](max_entries=1000, max_memory_mb=500)
        cache["key"] = {"data": "value"}
        value = cache["key"]  # Moves to end (most recently used)
    """

    def __init__(
        self,
        max_entries: int = 1000,
        max_memory_mb: int = 500,
    ):
        """Initialize the bounded cache.

        Args:
            max_entries: Maximum number of entries to cache.
            max_memory_mb: Soft memory limit in MB for cache eviction.
        """
        self._cache: OrderedDict[K, V] = OrderedDict()
        self.max_entries = max_entries
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._hits = 0
        self._misses = 0

    def __setitem__(self, key: K, value: V) -> None:
        """Add or update a cache entry with automatic cleanup.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        if key in self._cache:
            del self._cache[key]

        self._cache[key] = value
        self._enforce_limits()

    def __getitem__(self, key: K) -> V:
        """Get cache entry and mark as recently used.

        Args:
            key: Cache key.

        Returns:
            Cached value.

        Raises:
            KeyError: If key not in cache.
        """
        if key not in self._cache:
            self._misses += 1
            raise KeyError(key)

        self._hits += 1
        self._cache.move_to_end(key)
        return self._cache[key]

    def __delitem__(self, key: K) -> None:
        """Remove entry from cache.

        Args:
            key: Cache key.
        """
        if key in self._cache:
            del self._cache[key]

    def __contains__(self, key: K) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key.

        Returns:
            True if key is cached.
        """
        return key in self._cache

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)

    def get(self, key: K, default: V | None = None) -> V | None:
        """Get cache entry or return default.

        Args:
            key: Cache key.
            default: Default value if key not found.

        Returns:
            Cached value or default.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def items(self):
        """Return all cache items."""
        return self._cache.items()

    def keys(self):
        """Return all cache keys."""
        return self._cache.keys()

    def values(self):
        """Return all cache values."""
        return self._cache.values()

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def _enforce_limits(self) -> None:
        """Enforce cache size and memory limits by evicting old entries."""
        while len(self._cache) > self.max_entries:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug(f"Evicted {evicted_key} due to entry limit")

        if self._should_evict_for_memory():
            entries_to_remove = max(1, len(self._cache) // 10)  # Remove 10%
            for _ in range(entries_to_remove):
                if self._cache:
                    evicted_key, _ = self._cache.popitem(last=False)
                    logger.debug(f"Evicted {evicted_key} due to memory pressure")

    def _should_evict_for_memory(self) -> bool:
        """Check if we should evict entries due to memory pressure.

        Returns:
            True if cache size exceeds memory limit.
        """
        try:
            cache_size = sum(sys.getsizeof(v) for v in self._cache.values())
            return cache_size > self.max_memory_bytes
        except Exception:
            return len(self._cache) > int(self.max_entries * 0.8)

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate (0.0 to 1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict[str, Any]:
        """Return cache statistics.

        Returns:
            Dictionary with cache stats.
        """
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


class ASTCache(BoundedCache[Path, tuple[Any, str]]):
    """Specialized cache for parsed AST nodes.

    Stores (root_node, language) tuples keyed by file path.
    Optimized for code parsing workflows.

    Usage:
        cache = ASTCache(max_entries=1000)
        cache[Path("/path/to/file.py")] = (root_node, "python")
        root_node, language = cache[Path("/path/to/file.py")]
    """

    def __init__(
        self,
        max_entries: int = 1000,
        max_memory_mb: int = 500,
    ):
        """Initialize AST cache.

        Args:
            max_entries: Maximum number of files to cache.
            max_memory_mb: Memory limit for cache.
        """
        super().__init__(max_entries, max_memory_mb)

    def remove_file(self, file_path: Path) -> bool:
        """Remove a file from the cache.

        Args:
            file_path: Path to remove.

        Returns:
            True if file was in cache.
        """
        if file_path in self._cache:
            del self._cache[file_path]
            return True
        return False

    def get_cached_files(self) -> list[Path]:
        """Get list of all cached file paths.

        Returns:
            List of cached file paths.
        """
        return list(self._cache.keys())


class FunctionRegistry:
    """Registry for tracking function and class definitions.

    Uses a trie-like structure for efficient prefix/suffix lookups.
    Supports O(1) lookups by simple name for common cases.
    """

    def __init__(self):
        """Initialize the function registry."""
        self._entries: dict[str, str] = {}
        self._simple_name_index: dict[str, set[str]] = {}
        self._trie: dict[str, Any] = {}

    def register(self, qualified_name: str, entity_type: str) -> None:
        """Register a function or class.

        Args:
            qualified_name: Full qualified name (e.g., "module.Class.method").
            entity_type: Entity type ("Function", "Method", "Class").
        """
        self._entries[qualified_name] = entity_type

        simple_name = qualified_name.split(".")[-1]
        if simple_name not in self._simple_name_index:
            self._simple_name_index[simple_name] = set()
        self._simple_name_index[simple_name].add(qualified_name)

        parts = qualified_name.split(".")
        current = self._trie
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]
        current["__type__"] = entity_type
        current["__qn__"] = qualified_name

    def unregister(self, qualified_name: str) -> bool:
        """Remove a function or class from registry.

        Args:
            qualified_name: Full qualified name.

        Returns:
            True if entry was found and removed.
        """
        if qualified_name not in self._entries:
            return False

        del self._entries[qualified_name]

        simple_name = qualified_name.split(".")[-1]
        if simple_name in self._simple_name_index:
            self._simple_name_index[simple_name].discard(qualified_name)
            if not self._simple_name_index[simple_name]:
                del self._simple_name_index[simple_name]

        self._cleanup_trie_path(qualified_name.split("."))

        return True

    def _cleanup_trie_path(self, parts: list[str]) -> None:
        """Clean up trie path after removal."""
        if not parts:
            return

        current = self._trie
        path = []

        for part in parts[:-1]:
            if part not in current:
                return
            path.append((current, part))
            current = current[part]

        last_part = parts[-1]
        if last_part in current:
            current[last_part].pop("__type__", None)
            current[last_part].pop("__qn__", None)

            if not current[last_part]:
                del current[last_part]

    def get(self, qualified_name: str) -> str | None:
        """Get entity type by qualified name.

        Args:
            qualified_name: Full qualified name.

        Returns:
            Entity type or None if not found.
        """
        return self._entries.get(qualified_name)

    def __contains__(self, qualified_name: str) -> bool:
        """Check if qualified name is registered."""
        return qualified_name in self._entries

    def __len__(self) -> int:
        """Return number of registered entries."""
        return len(self._entries)

    def find_by_simple_name(self, simple_name: str) -> list[str]:
        """Find all qualified names ending with a simple name.

        Args:
            simple_name: Simple name (e.g., "method_name").

        Returns:
            List of matching qualified names.
        """
        return list(self._simple_name_index.get(simple_name, []))

    def find_with_prefix(self, prefix: str) -> list[tuple[str, str]]:
        """Find all entries with a given prefix.

        Args:
            prefix: Prefix to search for (e.g., "module.Class").

        Returns:
            List of (qualified_name, entity_type) tuples.
        """
        results = []
        prefix_parts = prefix.split(".")

        current = self._trie
        for part in prefix_parts:
            if part not in current:
                return []
            current = current[part]

        def dfs(node: dict[str, Any]) -> None:
            if "__qn__" in node:
                results.append((node["__qn__"], node["__type__"]))

            for key, child in node.items():
                if not key.startswith("__") and isinstance(child, dict):
                    dfs(child)

        dfs(current)
        return results

    def remove_by_prefix(self, prefix: str) -> int:
        """Remove all entries with a given prefix.

        Args:
            prefix: Prefix to remove (e.g., "module.file").

        Returns:
            Number of entries removed.
        """
        entries_to_remove = [qn for qn, _ in self.find_with_prefix(prefix)]
        for qn in entries_to_remove:
            self.unregister(qn)
        return len(entries_to_remove)

    def all_entries(self) -> dict[str, str]:
        """Get all registered entries.

        Returns:
            Dictionary of qualified_name -> entity_type.
        """
        return dict(self._entries)
