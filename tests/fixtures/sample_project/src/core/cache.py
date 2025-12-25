"""Multi-layer caching system with TTL and invalidation support.

This module provides a sophisticated caching infrastructure used
throughout the application for performance optimization.

Architecture:
    L1 Cache (Memory) -> L2 Cache (Redis) -> Database

Cache invalidation is event-driven via the EventBus.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Generic, TypeVar, Callable
from collections import OrderedDict
import asyncio
import hashlib
import json

from code_rag.tests.fixtures.sample_project.src.core.events import (
    EventBus,
    Event,
    EventType,
)


T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """Wrapper for cached values with metadata."""

    value: T
    created_at: datetime
    expires_at: datetime | None
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def record_hit(self) -> None:
        """Increment hit counter for analytics."""
        self.hit_count += 1


class CacheBackend(ABC):
    """Abstract interface for cache storage backends."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Retrieve value by key."""
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: timedelta | None = None,
    ) -> None:
        """Store value with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Remove key from cache."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Remove all entries."""
        pass


class MemoryCache(CacheBackend):
    """In-memory LRU cache for L1 caching.

    Uses OrderedDict to maintain LRU order with O(1) operations.
    Thread-safe through asyncio locks.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    async def get(self, key: str) -> Any | None:
        """Get value, updating LRU order."""
        async with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            entry = self._cache[key]
            if entry.is_expired:
                del self._cache[key]
                self._stats["misses"] += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.record_hit()
            self._stats["hits"] += 1
            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: timedelta | None = None,
    ) -> None:
        """Set value, evicting LRU if at capacity."""
        async with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                self._stats["evictions"] += 1

            expires_at = None
            if ttl:
                expires_at = datetime.utcnow() + ttl

            self._cache[key] = CacheEntry(
                value=value,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
            )

    async def delete(self, key: str) -> bool:
        """Remove key from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if non-expired key exists."""
        async with self._lock:
            if key not in self._cache:
                return False
            if self._cache[key].is_expired:
                del self._cache[key]
                return False
            return True

    async def clear(self) -> None:
        """Clear all entries."""
        async with self._lock:
            self._cache.clear()

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        return {
            **self._stats,
            "size": len(self._cache),
            "hit_rate": hit_rate,
        }


class RedisCache(CacheBackend):
    """Redis-based cache for L2 caching (distributed).

    Provides shared cache across application instances.
    Serializes values to JSON for storage.
    """

    def __init__(self, redis_client: Any, prefix: str = "cache:"):
        self.redis = redis_client
        self.prefix = prefix

    def _make_key(self, key: str) -> str:
        """Add prefix to key for namespacing."""
        return f"{self.prefix}{key}"

    async def get(self, key: str) -> Any | None:
        """Get and deserialize value."""
        data = await self.redis.get(self._make_key(key))
        if data is None:
            return None
        return json.loads(data)

    async def set(
        self,
        key: str,
        value: Any,
        ttl: timedelta | None = None,
    ) -> None:
        """Serialize and store value."""
        data = json.dumps(value)
        if ttl:
            await self.redis.setex(
                self._make_key(key),
                int(ttl.total_seconds()),
                data,
            )
        else:
            await self.redis.set(self._make_key(key), data)

    async def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        result = await self.redis.delete(self._make_key(key))
        return result > 0

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        return await self.redis.exists(self._make_key(key)) > 0

    async def clear(self) -> None:
        """Clear all keys with our prefix."""
        keys = await self.redis.keys(f"{self.prefix}*")
        if keys:
            await self.redis.delete(*keys)


class Cache:
    """Multi-layer cache with automatic fallback.

    Provides a unified interface for the caching system:
    1. Check L1 (memory) cache first
    2. Fall back to L2 (Redis) cache
    3. Populate upper layers on miss

    Integrates with EventBus for invalidation.
    """

    def __init__(
        self,
        l1_cache: MemoryCache | None = None,
        l2_cache: RedisCache | None = None,
        default_ttl: timedelta = timedelta(hours=1),
    ):
        self.l1 = l1_cache or MemoryCache()
        self.l2 = l2_cache
        self.default_ttl = default_ttl

    async def get(self, key: str) -> Any | None:
        """Get value from cache hierarchy."""
        # Try L1 first
        value = await self.l1.get(key)
        if value is not None:
            return value

        # Try L2 if available
        if self.l2:
            value = await self.l2.get(key)
            if value is not None:
                # Populate L1 from L2
                await self.l1.set(key, value, self.default_ttl)
                return value

        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: timedelta | None = None,
    ) -> None:
        """Set value in all cache layers."""
        ttl = ttl or self.default_ttl

        # Write to both layers
        await self.l1.set(key, value, ttl)
        if self.l2:
            await self.l2.set(key, value, ttl)

    async def invalidate(self, key: str) -> None:
        """Remove key from all cache layers and emit event."""
        await self.l1.delete(key)
        if self.l2:
            await self.l2.delete(key)

        # Emit cache invalidation event
        await EventBus.get_instance().publish(
            Event(
                type=EventType.CACHE_INVALIDATED,
                payload={"key": key},
                source="cache",
            )
        )

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        # This is simplified - real implementation would scan keys
        count = 0
        # In production, would iterate matching keys
        return count

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: timedelta | None = None,
    ) -> Any:
        """Get value or compute and cache it.

        This is the most common caching pattern - check cache,
        compute if missing, then cache the result.
        """
        value = await self.get(key)
        if value is not None:
            return value

        # Compute value
        result = factory()
        if asyncio.iscoroutine(result):
            value = await result
        else:
            value = result

        await self.set(key, value, ttl)
        return value


def cached(
    ttl: timedelta = timedelta(hours=1),
    key_prefix: str = "",
    cache_instance: Cache | None = None,
):
    """Decorator for caching function results.

    Automatically generates cache keys from function arguments.

    Example:
        @cached(ttl=timedelta(minutes=30), key_prefix="user")
        async def get_user(user_id: int) -> User:
            return await db.fetch_user(user_id)
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key from function name and arguments
            key_parts = [key_prefix or func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)

            # Hash long keys
            if len(cache_key) > 200:
                cache_key = hashlib.md5(cache_key.encode()).hexdigest()

            cache = cache_instance or _default_cache

            return await cache.get_or_set(
                cache_key,
                lambda: func(*args, **kwargs),
                ttl,
            )

        return wrapper
    return decorator


# Default cache instance
_default_cache: Cache | None = None


def init_cache(l1: MemoryCache | None = None, l2: RedisCache | None = None) -> Cache:
    """Initialize the default cache instance."""
    global _default_cache
    _default_cache = Cache(l1_cache=l1, l2_cache=l2)
    return _default_cache


def get_cache() -> Cache:
    """Get the default cache instance."""
    if _default_cache is None:
        return init_cache()
    return _default_cache
