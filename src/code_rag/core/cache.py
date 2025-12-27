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
    """Memory-aware LRU cache with automatic eviction."""

    def __init__(self, max_entries: int = 1000, max_memory_mb: int = 500):
        self._cache: OrderedDict[K, V] = OrderedDict()
        self.max_entries = max_entries
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._hits = 0
        self._misses = 0

    def __setitem__(self, key: K, value: V) -> None:
        if key in self._cache:
            del self._cache[key]
        self._cache[key] = value
        self._enforce_limits()

    def __getitem__(self, key: K) -> V:
        if key not in self._cache:
            self._misses += 1
            raise KeyError(key)
        self._hits += 1
        self._cache.move_to_end(key)
        return self._cache[key]

    def __delitem__(self, key: K) -> None:
        if key in self._cache:
            del self._cache[key]

    def __contains__(self, key: K) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def get(self, key: K, default: V | None = None) -> V | None:
        try:
            return self[key]
        except KeyError:
            return default

    def items(self):
        return self._cache.items()

    def keys(self):
        return self._cache.keys()

    def values(self):
        return self._cache.values()

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def _enforce_limits(self) -> None:
        while len(self._cache) > self.max_entries:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug(f"Evicted {evicted_key} due to entry limit")

        if self._should_evict_for_memory():
            entries_to_remove = max(1, len(self._cache) // 10)
            for _ in range(entries_to_remove):
                if self._cache:
                    evicted_key, _ = self._cache.popitem(last=False)
                    logger.debug(f"Evicted {evicted_key} due to memory pressure")

    def _should_evict_for_memory(self) -> bool:
        try:
            cache_size = sum(sys.getsizeof(v) for v in self._cache.values())
            return cache_size > self.max_memory_bytes
        except Exception:
            return len(self._cache) > int(self.max_entries * 0.8)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


class ASTCache(BoundedCache[Path, tuple[Any, str]]):
    """Cache for parsed AST nodes, stores (root_node, language) tuples."""

    def remove_file(self, file_path: Path) -> bool:
        if file_path in self._cache:
            del self._cache[file_path]
            return True
        return False

    def get_cached_files(self) -> list[Path]:
        return list(self._cache.keys())


class FunctionRegistry:
    """Registry for tracking function/class definitions with trie-based lookups."""

    def __init__(self):
        self._entries: dict[str, str] = {}
        self._simple_name_index: dict[str, set[str]] = {}
        self._trie: dict[str, Any] = {}

    def register(self, qualified_name: str, entity_type: str) -> None:
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
        return self._entries.get(qualified_name)

    def __contains__(self, qualified_name: str) -> bool:
        return qualified_name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def find_by_simple_name(self, simple_name: str) -> list[str]:
        return list(self._simple_name_index.get(simple_name, []))

    def find_with_prefix(self, prefix: str) -> list[tuple[str, str]]:
        results = []
        prefix_parts = prefix.split(".")

        current = self._trie
        for part in prefix_parts:
            if part not in current:
                return []
            current = current[part]

        def collect_entries(node: dict[str, Any]) -> None:
            if "__qn__" in node:
                results.append((node["__qn__"], node["__type__"]))
            for key, child in node.items():
                if not key.startswith("__") and isinstance(child, dict):
                    collect_entries(child)

        collect_entries(current)
        return results

    def remove_by_prefix(self, prefix: str) -> int:
        entries_to_remove = [qn for qn, _ in self.find_with_prefix(prefix)]
        for qn in entries_to_remove:
            self.unregister(qn)
        return len(entries_to_remove)

    def all_entries(self) -> dict[str, str]:
        return dict(self._entries)
