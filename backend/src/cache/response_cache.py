"""Answer cache in front of the graph: exact-match and semantic.

Why it is safe
--------------
The cache key is the *access scope*, not just the question. Two callers with
different roles, departments or document scopes never share an entry, so a
store associate can never be served a compliance officer's answer. Only
released answers (``status == "answered"``) that were certified at or above the
confidence threshold, were not degraded and hit no budget stop are stored;
escalations, refusals and clarifications are never cached. Follow-up turns in
a conversation are not looked up either, because the meaning of "and for the
other vendor?" depends on the thread, not on the words.

Exact vs. semantic
------------------
An exact hit compares a normalised form of the question (lower-cased,
whitespace collapsed, trailing punctuation dropped). A semantic hit embeds the
question with the same Azure embedding deployment the vector index uses and
accepts the nearest cached question in the same scope when its cosine
similarity is at least ``SEMANTIC_CACHE_THRESHOLD``. Embeddings are cached
too, so a lookup that misses and then stores does not pay for the embedding
twice. When Azure is not configured, or the embedding call fails, the cache
silently degrades to exact-match only and says so in the log.

What a hit saves
----------------
Each entry remembers the wall-clock and token cost of the run that produced
it, so the metrics report can state how much latency and how many tokens the
cache has returned to the caller, not just a hit rate.
"""

import hashlib
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from configs.settings import settings
from src.cache.base import TTLCache

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")
_TRAILING_PUNCTUATION = re.compile(r"[\s?.!]+$")

EmbedFn = Callable[[str], list[float]]


def normalise_query(query: str) -> str:
    text = _WHITESPACE.sub(" ", (query or "").strip().lower())

    return _TRAILING_PUNCTUATION.sub("", text)


def scope_key(
    role: str,
    access_scopes: list[str],
    departments: list[str] | None = None,
    document_scope: list[str] | None = None,
    as_of: str | None = None,
) -> str:
    """Everything that can change the *answer* for the same words, hashed."""
    material = "|".join(
        [
            role or "",
            ",".join(sorted(access_scopes or [])),
            ",".join(sorted(departments or [])),
            ",".join(sorted(document_scope or [])),
            as_of or str(settings.as_of_date),
        ]
    )

    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def exact_key(scope: str, query: str) -> str:
    digest = hashlib.sha256(normalise_query(query).encode("utf-8")).hexdigest()[:32]

    return f"{scope}:{digest}"


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    try:
        import numpy as np

        a = np.asarray(left, dtype=float)
        b = np.asarray(right, dtype=float)
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))

        return float(a.dot(b) / denominator) if denominator else 0.0
    except ImportError:  # pragma: no cover - numpy is a transitive dependency
        dot = sum(x * y for x, y in zip(left, right, strict=True))
        norm_l = sum(x * x for x in left) ** 0.5
        norm_r = sum(y * y for y in right) ** 0.5

        return dot / (norm_l * norm_r) if norm_l and norm_r else 0.0


@dataclass(slots=True)
class CachedAnswer:
    body: dict
    status_code: int
    scope: str
    query: str
    normalised_query: str
    embedding: list[float] | None
    source_request_id: str
    produced_in_ms: float
    tokens_spent: int
    evidence_path: str | None = None
    confidence: float | None = None


@dataclass(slots=True)
class CacheHit:
    entry: CachedAnswer
    kind: str
    similarity: float
    age_seconds: float
    matched_query: str


@dataclass(slots=True)
class LookupResult:
    hit: CacheHit | None
    embedding: list[float] | None = None
    reason: str = ""


@dataclass(slots=True)
class ResponseCacheStats:
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    skipped: int = 0
    stores: int = 0
    rejected_stores: int = 0
    embedding_failures: int = 0
    saved_ms: float = 0.0
    saved_tokens: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        lookups = self.exact_hits + self.semantic_hits + self.misses

        return {
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "hits": self.exact_hits + self.semantic_hits,
            "misses": self.misses,
            "lookups": lookups,
            "hit_rate": round((self.exact_hits + self.semantic_hits) / lookups, 4) if lookups else 0.0,
            "skipped_lookups": self.skipped,
            "skip_reasons": dict(self.skip_reasons),
            "stores": self.stores,
            "rejected_stores": self.rejected_stores,
            "embedding_failures": self.embedding_failures,
            "saved_ms_total": round(self.saved_ms, 2),
            "saved_tokens_total": self.saved_tokens,
        }


def _default_embedder() -> EmbedFn | None:
    """The project's Azure embedding deployment, or ``None`` when not configured."""
    if not settings.azure_configured:
        return None

    from configs.llms import get_embedding_model

    model = get_embedding_model()

    return model.embed_query


class ResponseCache:
    def __init__(
        self,
        max_entries: int | None = None,
        ttl_seconds: float | None = None,
        semantic_threshold: float | None = None,
        embedder: EmbedFn | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self._store: TTLCache[str, CachedAnswer] = TTLCache(
            "response",
            max_entries=max_entries or settings.response_cache_max_entries,
            ttl_seconds=ttl_seconds or settings.response_cache_ttl_seconds,
            clock=clock,
        )
        self._embeddings: TTLCache[str, list[float]] = TTLCache(
            "query_embedding",
            max_entries=(max_entries or settings.response_cache_max_entries) * 2,
            ttl_seconds=(ttl_seconds or settings.response_cache_ttl_seconds) * 2,
            clock=clock,
        )
        self.semantic_threshold = (
            semantic_threshold if semantic_threshold is not None else settings.semantic_cache_threshold
        )
        self._embedder = embedder
        self._embedder_resolved = embedder is not None
        self._lock = threading.RLock()
        self.stats = ResponseCacheStats()

    def _resolve_embedder(self) -> EmbedFn | None:
        if not self._embedder_resolved:
            try:
                self._embedder = _default_embedder()
            except Exception as exc:
                logger.warning("semantic cache: embedding model unavailable (%s); exact-match only", exc)
                self._embedder = None

            self._embedder_resolved = True

            if self._embedder is None:
                logger.info("semantic cache disabled: no embedding model configured; exact-match only")

        return self._embedder

    def embed(self, query: str) -> list[float] | None:
        if not settings.enable_semantic_cache or self.semantic_threshold >= 1.0:
            return None

        embedder = self._resolve_embedder()

        if embedder is None:
            return None

        key = normalise_query(query)
        cached = self._embeddings.get(key)

        if cached is not None:
            return cached

        started = time.monotonic()

        try:
            vector = list(embedder(query))
        except Exception as exc:
            self.stats.embedding_failures += 1
            logger.warning("semantic cache: embedding failed (%s); falling back to exact match", exc)

            return None

        self._embeddings.put(key, vector)
        logger.debug(
            "semantic cache: embedded query dims=%d elapsed_ms=%.1f",
            len(vector),
            (time.monotonic() - started) * 1000,
        )

        return vector

    def skip(self, reason: str) -> None:
        with self._lock:
            self.stats.skipped += 1
            self.stats.skip_reasons[reason] = self.stats.skip_reasons.get(reason, 0) + 1

        logger.debug("response cache: lookup skipped reason=%s", reason)

    def lookup(self, scope: str, query: str) -> LookupResult:
        entry = self._store.get_entry(exact_key(scope, query))

        if entry is not None:
            with self._lock:
                self.stats.exact_hits += 1
                self.stats.saved_ms += entry.value.produced_in_ms
                self.stats.saved_tokens += entry.value.tokens_spent

            hit = CacheHit(
                entry=entry.value,
                kind="exact",
                similarity=1.0,
                age_seconds=self._store.age_seconds(entry),
                matched_query=entry.value.query,
            )
            logger.info(
                "response cache HIT kind=exact age_s=%.0f source_request_id=%s saved_ms=%.0f saved_tokens=%d",
                hit.age_seconds,
                entry.value.source_request_id,
                entry.value.produced_in_ms,
                entry.value.tokens_spent,
            )

            return LookupResult(hit=hit)

        embedding = self.embed(query)

        if embedding is not None:
            best = self._nearest(scope, embedding)

            if best is not None:
                _, candidate, similarity = best

                if similarity >= self.semantic_threshold:
                    with self._lock:
                        self.stats.semantic_hits += 1
                        self.stats.saved_ms += candidate.value.produced_in_ms
                        self.stats.saved_tokens += candidate.value.tokens_spent

                    candidate.hits += 1
                    hit = CacheHit(
                        entry=candidate.value,
                        kind="semantic",
                        similarity=round(similarity, 4),
                        age_seconds=self._store.age_seconds(candidate),
                        matched_query=candidate.value.query,
                    )
                    logger.info(
                        "response cache HIT kind=semantic similarity=%.4f threshold=%.2f age_s=%.0f "
                        "source_request_id=%s matched=%r",
                        similarity,
                        self.semantic_threshold,
                        hit.age_seconds,
                        candidate.value.source_request_id,
                        _preview(candidate.value.query),
                    )

                    return LookupResult(hit=hit, embedding=embedding)

                logger.debug(
                    "response cache: nearest neighbour similarity=%.4f below threshold=%.2f",
                    similarity,
                    self.semantic_threshold,
                )

        with self._lock:
            self.stats.misses += 1

        logger.info("response cache MISS scope=%s semantic=%s", scope, embedding is not None)

        return LookupResult(hit=None, embedding=embedding, reason="miss")

    def _nearest(self, scope: str, embedding: list[float]):
        best_key = None
        best_entry = None
        best_score = -1.0

        for key, entry in self._store.live_entries():
            value = entry.value

            if value.scope != scope or value.embedding is None:
                continue

            score = cosine_similarity(embedding, value.embedding)

            if score > best_score:
                best_key, best_entry, best_score = key, entry, score

        if best_entry is None:
            return None

        return best_key, best_entry, best_score

    def put(
        self,
        scope: str,
        query: str,
        body: dict,
        status_code: int,
        *,
        source_request_id: str,
        produced_in_ms: float,
        tokens_spent: int,
        embedding: list[float] | None = None,
        evidence_path: str | None = None,
        confidence: float | None = None,
    ) -> bool:
        if embedding is None:
            embedding = self.embed(query)

        entry = CachedAnswer(
            body=body,
            status_code=status_code,
            scope=scope,
            query=query,
            normalised_query=normalise_query(query),
            embedding=embedding,
            source_request_id=source_request_id,
            produced_in_ms=float(produced_in_ms),
            tokens_spent=int(tokens_spent),
            evidence_path=evidence_path,
            confidence=confidence,
        )

        self._store.put(exact_key(scope, query), entry)

        with self._lock:
            self.stats.stores += 1

        logger.info(
            "response cache STORE request_id=%s path=%s confidence=%s produced_in_ms=%.0f semantic=%s entries=%d",
            source_request_id,
            evidence_path,
            confidence,
            produced_in_ms,
            embedding is not None,
            len(self._store),
        )

        return True

    def reject(self, reason: str) -> None:
        with self._lock:
            self.stats.rejected_stores += 1

        logger.debug("response cache: store rejected reason=%s", reason)

    def invalidate(self, reason: str = "") -> int:
        cleared = self._store.clear(reason)
        self._embeddings.clear(reason)

        return cleared

    def __len__(self) -> int:
        return len(self._store)

    def report(self) -> dict[str, Any]:
        return {
            **self._store.report(),
            **self.stats.as_dict(),
            "semantic_enabled": bool(settings.enable_semantic_cache and self.semantic_threshold < 1.0),
            "semantic_threshold": self.semantic_threshold,
            "embedding_cache": self._embeddings.report(),
        }


def _preview(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


response_cache = ResponseCache()


def get_response_cache() -> ResponseCache:
    return response_cache


def should_cache_answer(body: dict) -> tuple[bool, str]:
    """Decide whether a released ``/ask`` body is safe to serve again.

    Returns ``(True, "")`` or ``(False, reason)``.
    """
    if body.get("status") != "answered":
        return False, f"status={body.get('status')}"

    if body.get("not_found"):
        # an honest "I don't know" is not a certified answer: the corpus or the records may answer it
        # on the next ask, and one model's "not in the extracts" must not be replayed for an hour
        return False, "not_found"

    if body.get("degraded"):
        return False, "degraded"

    threshold = settings.confidence_threshold
    confidence = float(body.get("confidence") or 0.0)

    if confidence < threshold:
        return False, f"confidence {confidence} below {threshold}"

    trace = body.get("trace") or {}

    if trace.get("budget_stops"):
        return False, "budget_stops"

    if trace.get("skipped_optional_nodes"):
        return False, "skipped_optional_nodes"

    return True, ""
