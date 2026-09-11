import hashlib
import json
import logging
import time
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)

IDEMPOTENCY_TTL_SECONDS = 900


@dataclass(slots=True)
class CachedResponse:
    payload: dict
    status_code: int
    stored_at: float


class IdempotencyStore:
    def __init__(self, ttl_seconds: int = IDEMPOTENCY_TTL_SECONDS):
        self._entries: dict[str, CachedResponse] = {}
        self._lock = Lock()
        self._ttl = ttl_seconds

    def _prune(self) -> None:
        cutoff = time.time() - self._ttl
        stale = [key for key, entry in self._entries.items() if entry.stored_at < cutoff]

        for key in stale:
            self._entries.pop(key, None)

        if stale:
            logger.debug("idempotency store pruned %d expired entr%s", len(stale), "y" if len(stale) == 1 else "ies")

    def get(self, key: str) -> CachedResponse | None:
        with self._lock:
            self._prune()
            entry = self._entries.get(key)

        if entry is not None:
            logger.info(
                "idempotent replay key=%s status=%s age_s=%.0f",
                key,
                entry.status_code,
                time.time() - entry.stored_at,
            )

        return entry

    def put(self, key: str, payload: dict, status_code: int) -> None:
        with self._lock:
            self._entries[key] = CachedResponse(payload=payload, status_code=status_code, stored_at=time.time())
            size = len(self._entries)

        logger.debug("idempotency store put key=%s status=%s entries=%d", key, status_code, size)


def build_key(user_id: str, idempotency_header: str | None, body: dict) -> str:
    if idempotency_header:
        return f"{user_id}:{idempotency_header}"

    fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:32]

    return f"{user_id}:auto:{fingerprint}"


idempotency_store = IdempotencyStore()
