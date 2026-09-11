"""Cost and latency optimisation: caches.

Three caches sit at three different depths of a request, each cutting a
different cost:

* :mod:`src.cache.response_cache` - in front of the graph. A hit returns a
  previously certified answer for the same question (exact or semantic match)
  in the same access scope, skipping every model call and every database read.
* :mod:`src.cache.retrieval_cache` - inside the RAG leg. The same query in the
  same document scope reuses the fused and reranked chunk list, so the
  embedding call, the BM25 pass and the cross-encoder are not repeated.
* :mod:`src.cache.llm_cache` - at the LangChain model boundary. An identical
  prompt to an identical model configuration returns the stored completion.

All three are process-local, bounded (LRU) and time-limited (TTL), and every
one of them reports hits, misses and the tokens / milliseconds it saved through
:func:`src.cache.registry.cache_report` (``GET /metrics/optimization``).
"""

from src.cache.base import CacheStats, TTLCache

__all__ = ["CacheStats", "TTLCache"]
