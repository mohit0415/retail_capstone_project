"""A small thread-safe TTL + LRU cache with hit/miss accounting.

Every cache in :mod:`src.cache` is built on this. It is deliberately
dependency-free so the unit tests can exercise it with a fake clock.
"""

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    puts: int = 0
    evictions: int = 0
    expirations: int = 0
    invalidations: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        total = self.lookups

        return round(self.hits / total, 4) if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "lookups": self.lookups,
            "hit_rate": self.hit_rate,
            "puts": self.puts,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "invalidations": self.invalidations,
        }


@dataclass(slots=True)
class _Entry[V]:
    value: V
    stored_at: float
    expires_at: float
    hits: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


class TTLCache[K, V]:
    """Bounded LRU cache whose entries expire ``ttl_seconds`` after they were stored.

    ``clock`` is injectable so tests can move time forward without sleeping.
    ``name`` only labels log lines and the metrics report.
    """

    def __init__(
        self,
        name: str,
        max_entries: int = 500,
        ttl_seconds: float = 3600.0,
        clock: Callable[[], float] = time.time,
    ):
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")

        self.name = name
        self.max_entries = int(max_entries)
        self.ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._entries: OrderedDict[K, _Entry[V]] = OrderedDict()
        self._lock = threading.RLock()
        self.stats = CacheStats()

    def get(self, key: K) -> V | None:
        entry = self.get_entry(key)

        return entry.value if entry is not None else None

    def get_entry(self, key: K) -> _Entry[V] | None:
        """Like :meth:`get` but returns the entry so callers can read age / meta."""
        now = self._clock()

        with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                self.stats.misses += 1

                return None

            if entry.expires_at <= now:
                del self._entries[key]
                self.stats.expirations += 1
                self.stats.misses += 1
                logger.debug("cache=%s expired key=%s", self.name, _short(key))

                return None

            self._entries.move_to_end(key)
            entry.hits += 1
            self.stats.hits += 1

            return entry

    def put(
        self, key: K, value: V, meta: dict[str, Any] | None = None, ttl_seconds: float | None = None
    ) -> None:
        now = self._clock()
        ttl = self.ttl_seconds if ttl_seconds is None else float(ttl_seconds)

        with self._lock:
            if key in self._entries:
                del self._entries[key]

            self._entries[key] = _Entry(
                value=value, stored_at=now, expires_at=now + ttl, meta=dict(meta or {})
            )
            self.stats.puts += 1

            while len(self._entries) > self.max_entries:
                evicted_key, _ = self._entries.popitem(last=False)
                self.stats.evictions += 1
                logger.debug(
                    "cache=%s evicted key=%s (lru, max_entries=%d)",
                    self.name,
                    _short(evicted_key),
                    self.max_entries,
                )

    def delete(self, key: K) -> bool:
        with self._lock:
            return self._entries.pop(key, None) is not None

    def clear(self, reason: str = "") -> int:
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self.stats.invalidations += 1

        if count or reason:
            logger.info("cache=%s cleared entries=%d reason=%s", self.name, count, reason or "unspecified")

        return count

    def prune(self) -> int:
        """Drop every expired entry. Called lazily by :meth:`size` and the report."""
        now = self._clock()

        with self._lock:
            stale = [key for key, entry in self._entries.items() if entry.expires_at <= now]

            for key in stale:
                del self._entries[key]

            self.stats.expirations += len(stale)

        return len(stale)

    def __contains__(self, key: K) -> bool:
        with self._lock:
            entry = self._entries.get(key)

            return entry is not None and entry.expires_at > self._clock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def size(self) -> int:
        self.prune()

        return len(self)

    def live_entries(self) -> Iterator[tuple[K, _Entry[V]]]:
        """Snapshot of non-expired entries (a list copy, safe to iterate without the lock)."""
        now = self._clock()

        with self._lock:
            return iter([(key, entry) for key, entry in self._entries.items() if entry.expires_at > now])

    def age_seconds(self, entry: _Entry[V]) -> float:
        return round(max(0.0, self._clock() - entry.stored_at), 3)

    def report(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entries": self.size(),
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
            **self.stats.as_dict(),
        }


def _short(key: Any) -> str:
    text = str(key)

    return text if len(text) <= 48 else text[:45] + "..."
