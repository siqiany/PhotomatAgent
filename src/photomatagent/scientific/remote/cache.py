"""TTL cache for expensive, read-only SCNet capability probes.

Capability probing (connection check, partition discovery, module/executable
availability, POTCAR layout checks) can take on the order of a minute on
SCNet's first connection. Repeated model-tool calls must not re-run those
probes on every invocation: results are cached for a bounded TTL and keyed by
the probe identity (including its arguments), so different modules or
partitions never share entries.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable


class CapabilityCache:
    """In-memory, TTL-bounded cache of async probe results.

    ``get_or_call`` memoizes ``factory`` by ``key`` for ``ttl_seconds``.
    Failed probes (an exception from ``factory``) are NOT cached: a transient
    SSH failure must not poison the cache for the whole TTL window.
    """

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.ttl_seconds = float(ttl_seconds)
        self._store: dict[str, tuple[float, Any]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._gate = asyncio.Lock()
        self.hits = 0
        self.misses = 0

    async def get_or_call(
        self,
        key: str,
        factory: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Return a cached value, or compute, store and return it."""
        now = time.monotonic()
        entry = self._store.get(key)
        if entry is not None and now - entry[0] < self.ttl_seconds:
            self.hits += 1
            return entry[1]
        async with self._gate:
            lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            # Re-check under the per-key lock: another caller may have filled
            # the entry while we waited.
            entry = self._store.get(key)
            if entry is not None and time.monotonic() - entry[0] < self.ttl_seconds:
                self.hits += 1
                return entry[1]
            self.misses += 1
            value = await factory()
            self._store[key] = (time.monotonic(), value)
            return value

    def invalidate(self, key: str) -> None:
        """Drop one cached entry (uncached keys are a no-op)."""
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "entries": len(self._store),
            "ttl_seconds": self.ttl_seconds,
        }
