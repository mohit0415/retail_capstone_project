"""Retrieval-result cache for the policy corpus.

``retrieve_policy_evidence`` costs an embedding call, a vector search, a BM25
pass, reciprocal-rank fusion and (when enabled) a cross-encoder rerank. The
hybrid path, the high-risk panel and a reflection re-plan all ask for the
same query in the same scope within one request, and repeated questions ask
for it across requests. This cache returns the finished chunk list instead.

The key carries everything that changes the result: the query, the document
types the role may read, the requested scope, the as-of date, the candidate
count, the kept count and whether the reranker ran. Chunks are pydantic
models that later stages mutate (``rerank_score`` is assigned in place), so
the cache hands out deep copies and stores deep copies.

Entries are dropped whenever the corpus changes (``POST /ingest``,
``POST /ingest/rebuild``) via :func:`src.cache.registry.invalidate_corpus_caches`.
"""

import logging
import time
from datetime import date
from typing import Any

from configs.settings import settings
from src.cache.base import TTLCache
from src.schemas.models import RetrievedChunk

logger = logging.getLogger(__name__)

RetrievalValue = tuple[list[RetrievedChunk], bool, list[str], list[RetrievedChunk]]


def retrieval_key(
    query: str,
    allowed_doc_types: list[str],
    doc_scope: list[str] | None,
    as_of: date | str | None,
    top_k: int | None,
    top_n: int | None,
    use_rerank: bool,
) -> tuple:
    return (
        " ".join((query or "").split()).lower(),
        tuple(sorted(allowed_doc_types or [])),
        tuple(sorted(doc_scope or [])),
        str(as_of or settings.as_of_date),
        int(top_k or settings.retrieval_top_k),
        int(top_n or settings.rerank_top_n),
        bool(use_rerank),
    )


class RetrievalCache:
    def __init__(self, max_entries: int | None = None, ttl_seconds: float | None = None):
        self._store: TTLCache[tuple, RetrievalValue] = TTLCache(
            "retrieval",
            max_entries=max_entries or settings.retrieval_cache_max_entries,
            ttl_seconds=ttl_seconds or settings.retrieval_cache_ttl_seconds,
        )
        self.saved_ms = 0.0

    def get(self, key: tuple) -> RetrievalValue | None:
        entry = self._store.get_entry(key)

        if entry is None:
            return None

        chunks, fused, skipped, candidates = entry.value
        self.saved_ms += float(entry.meta.get("elapsed_ms", 0.0))

        logger.info(
            "retrieval cache HIT chunks=%d age_s=%.0f saved_ms=%.0f query=%r",
            len(chunks),
            self._store.age_seconds(entry),
            entry.meta.get("elapsed_ms", 0.0),
            _preview(key[0]),
        )

        return (
            [chunk.model_copy(deep=True) for chunk in chunks],
            fused,
            list(skipped),
            [chunk.model_copy(deep=True) for chunk in candidates],
        )

    def put(self, key: tuple, value: RetrievalValue, elapsed_ms: float) -> None:
        chunks, fused, skipped, candidates = value

        if not chunks:
            logger.debug("retrieval cache: not storing an empty result for %r", _preview(key[0]))

            return

        self._store.put(
            key,
            (
                [chunk.model_copy(deep=True) for chunk in chunks],
                fused,
                list(skipped),
                [chunk.model_copy(deep=True) for chunk in candidates],
            ),
            meta={"elapsed_ms": float(elapsed_ms)},
        )
        logger.debug(
            "retrieval cache STORE chunks=%d elapsed_ms=%.0f query=%r",
            len(chunks),
            elapsed_ms,
            _preview(key[0]),
        )

    def invalidate(self, reason: str = "") -> int:
        return self._store.clear(reason)

    def __len__(self) -> int:
        return len(self._store)

    def report(self) -> dict[str, Any]:
        return {**self._store.report(), "saved_ms_total": round(self.saved_ms, 2)}


retrieval_cache = RetrievalCache()


def timed() -> float:
    return time.monotonic()


def _preview(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."
