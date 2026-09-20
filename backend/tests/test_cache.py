"""Cost & latency: the three caches (src/cache).

Everything runs without Azure: the response cache gets a fake embedder, the
LLM cache a fake chat model, the retrieval cache plain chunk objects.
"""

import math

import pytest
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from configs.settings import settings
from src.cache import registry
from src.cache.base import TTLCache
from src.cache.llm_cache import TTLLLMCache, _model_from
from src.cache.response_cache import (
    ResponseCache,
    cosine_similarity,
    exact_key,
    normalise_query,
    scope_key,
    should_cache_answer,
)
from src.cache.retrieval_cache import RetrievalCache, retrieval_key
from src.schemas.models import RetrievedChunk


class Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_ttl_cache_expires_entries_after_the_ttl():
    clock = Clock()
    cache: TTLCache[str, str] = TTLCache("t", max_entries=10, ttl_seconds=60, clock=clock)

    cache.put("a", "1")
    assert cache.get("a") == "1"

    clock.now += 61
    assert cache.get("a") is None
    assert cache.stats.expirations == 1
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1


def test_ttl_cache_evicts_least_recently_used_when_full():
    cache: TTLCache[str, str] = TTLCache("t", max_entries=2, ttl_seconds=60, clock=Clock())

    cache.put("a", "1")
    cache.put("b", "2")
    assert cache.get("a") == "1"
    cache.put("c", "3")

    assert "b" not in cache
    assert cache.get("a") == "1"
    assert cache.get("c") == "3"
    assert cache.stats.evictions == 1


def test_ttl_cache_report_and_clear():
    cache: TTLCache[str, str] = TTLCache("t", max_entries=5, ttl_seconds=60, clock=Clock())
    cache.put("a", "1")
    cache.get("a")
    cache.get("zzz")

    report = cache.report()
    assert report["name"] == "t"
    assert report["entries"] == 1
    assert report["hit_rate"] == 0.5

    assert cache.clear("test") == 1
    assert cache.size() == 0
    assert cache.stats.invalidations == 1


def test_ttl_cache_rejects_a_zero_bound():
    with pytest.raises(ValueError):
        TTLCache("t", max_entries=0)


def test_normalise_query_ignores_case_whitespace_and_trailing_punctuation():
    assert normalise_query("  What IS the retention   period? ") == "what is the retention period"
    assert exact_key("s", "What is the retention period?") == exact_key("s", "what is the retention period")


def test_scope_key_changes_with_role_scopes_departments_and_document_scope():
    base = scope_key("compliance_officer", ["doc:gdpr", "table:vendors"], ["legal"], [], "2025-12-31")

    assert scope_key("compliance_officer", ["table:vendors", "doc:gdpr"], ["legal"], [], "2025-12-31") == base
    assert scope_key("store_associate", ["doc:gdpr", "table:vendors"], ["legal"], [], "2025-12-31") != base
    assert scope_key("compliance_officer", ["doc:gdpr"], ["legal"], [], "2025-12-31") != base
    assert scope_key("compliance_officer", ["doc:gdpr", "table:vendors"], ["it"], [], "2025-12-31") != base
    assert (
        scope_key("compliance_officer", ["doc:gdpr", "table:vendors"], ["legal"], ["gdpr"], "2025-12-31")
        != base
    )


def test_cosine_similarity():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([1.0, 1.0], [1.0]) == 0.0


_VECTORS = {
    "what is the retention period for invoices": [1.0, 0.0, 0.0],
    "how long are invoices retained": [0.98, 0.199, 0.0],
    "can we share customer data with an overseas vendor": [0.0, 0.0, 1.0],
}


def _embedder(text: str) -> list[float]:
    return list(_VECTORS[normalise_query(text)])


def _answer(request_id: str = "req-src") -> dict:
    return {
        "status": "answered",
        "request_id": request_id,
        "answer": "Seven years [Retention Policy §4.2].",
        "confidence": 0.9,
        "degraded": False,
        "trace": {"budget_stops": [], "skipped_optional_nodes": []},
    }


@pytest.fixture
def cache(monkeypatch):
    monkeypatch.setattr(settings, "enable_semantic_cache", True)

    return ResponseCache(
        max_entries=10, ttl_seconds=60, semantic_threshold=0.95, embedder=_embedder, clock=Clock()
    )


def test_exact_hit_returns_the_stored_answer_and_counts_savings(cache):
    scope = "scope-a"
    cache.put(
        scope,
        "What is the retention period for invoices?",
        _answer(),
        200,
        source_request_id="req-src",
        produced_in_ms=4200.0,
        tokens_spent=9000,
    )

    result = cache.lookup(scope, "what is the retention period for invoices")

    assert result.hit is not None
    assert result.hit.kind == "exact"
    assert result.hit.similarity == 1.0
    assert result.hit.entry.source_request_id == "req-src"
    assert cache.stats.exact_hits == 1
    assert cache.stats.saved_ms == 4200.0
    assert cache.stats.saved_tokens == 9000


def test_semantic_hit_when_the_nearest_cached_question_clears_the_threshold(cache):
    scope = "scope-a"
    cache.put(
        scope,
        "What is the retention period for invoices?",
        _answer(),
        200,
        source_request_id="req-src",
        produced_in_ms=4200.0,
        tokens_spent=9000,
    )

    result = cache.lookup(scope, "How long are invoices retained?")

    assert result.hit is not None
    assert result.hit.kind == "semantic"
    assert 0.95 <= result.hit.similarity <= 1.0
    assert result.hit.matched_query == "What is the retention period for invoices?"
    assert cache.stats.semantic_hits == 1


def test_no_hit_below_the_threshold_and_the_embedding_is_returned_for_reuse(cache):
    scope = "scope-a"
    cache.put(
        scope,
        "What is the retention period for invoices?",
        _answer(),
        200,
        source_request_id="req-src",
        produced_in_ms=1.0,
        tokens_spent=1,
    )

    result = cache.lookup(scope, "Can we share customer data with an overseas vendor?")

    assert result.hit is None
    assert result.embedding == [0.0, 0.0, 1.0]
    assert cache.stats.misses == 1


def test_entries_never_cross_an_access_scope(cache):
    cache.put(
        "officer",
        "What is the retention period for invoices?",
        _answer(),
        200,
        source_request_id="req-src",
        produced_in_ms=1.0,
        tokens_spent=1,
    )

    assert cache.lookup("associate", "What is the retention period for invoices?").hit is None
    assert cache.lookup("associate", "How long are invoices retained?").hit is None


def test_a_threshold_of_one_disables_semantic_matching(monkeypatch):
    monkeypatch.setattr(settings, "enable_semantic_cache", True)
    exact_only = ResponseCache(
        max_entries=10, ttl_seconds=60, semantic_threshold=1.0, embedder=_embedder, clock=Clock()
    )
    exact_only.put(
        "s",
        "What is the retention period for invoices?",
        _answer(),
        200,
        source_request_id="r",
        produced_in_ms=1.0,
        tokens_spent=1,
    )

    result = exact_only.lookup("s", "How long are invoices retained?")

    assert result.hit is None
    assert result.embedding is None


def test_an_embedding_failure_degrades_to_exact_match_only(monkeypatch):
    monkeypatch.setattr(settings, "enable_semantic_cache", True)

    def _boom(text: str) -> list[float]:
        raise RuntimeError("azure down")

    fragile = ResponseCache(
        max_entries=10, ttl_seconds=60, semantic_threshold=0.9, embedder=_boom, clock=Clock()
    )
    fragile.put(
        "s",
        "What is the retention period for invoices?",
        _answer(),
        200,
        source_request_id="r",
        produced_in_ms=1.0,
        tokens_spent=1,
    )

    assert fragile.lookup("s", "what is the retention period for invoices").hit.kind == "exact"
    assert fragile.lookup("s", "How long are invoices retained?").hit is None
    assert fragile.stats.embedding_failures >= 1


def test_entries_expire_with_the_ttl(monkeypatch):
    monkeypatch.setattr(settings, "enable_semantic_cache", False)
    clock = Clock()
    short = ResponseCache(max_entries=10, ttl_seconds=30, semantic_threshold=1.0, embedder=None, clock=clock)
    short.put("s", "q one", _answer(), 200, source_request_id="r", produced_in_ms=1.0, tokens_spent=1)

    clock.now += 31

    assert short.lookup("s", "q one").hit is None


def test_invalidate_empties_the_cache_and_the_report_reflects_it(cache):
    cache.put(
        "s",
        "What is the retention period for invoices?",
        _answer(),
        200,
        source_request_id="r",
        produced_in_ms=1.0,
        tokens_spent=1,
    )

    assert cache.invalidate("test") == 1
    assert len(cache) == 0

    report = cache.report()
    assert report["entries"] == 0
    assert report["stores"] == 1
    assert report["semantic_enabled"] is True
    assert "embedding_cache" in report


def test_skip_reasons_are_counted(cache):
    cache.skip("conversation_history")
    cache.skip("conversation_history")
    cache.skip("caller_opted_out")

    assert cache.report()["skip_reasons"] == {"conversation_history": 2, "caller_opted_out": 1}


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (_answer(), True),
        ({**_answer(), "status": "pending_review"}, False),
        ({**_answer(), "degraded": True}, False),
        ({**_answer(), "confidence": 0.2}, False),
        ({**_answer(), "trace": {"budget_stops": [{"node": "rag_path"}]}}, False),
        ({**_answer(), "trace": {"skipped_optional_nodes": ["reranker"]}}, False),
    ],
)
def test_only_clean_certified_answers_are_cacheable(body, expected):
    ok, reason = should_cache_answer(body)

    assert ok is expected
    assert (reason == "") is expected


def _chunk(clause: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c-{clause}",
        doc_type="retention_policy",
        document_title="Retention Policy",
        section="Retention",
        clause_number=clause,
        version="2.0",
        content="Records are retained for seven years.",
        dense_score=0.5,
    )


def test_retrieval_key_is_order_insensitive_and_normalises_the_query():
    a = retrieval_key(
        "What is  the period?", ["gdpr", "retention_policy"], ["gdpr"], "2025-12-31", 20, 6, True
    )
    b = retrieval_key(
        "what is the period?", ["retention_policy", "gdpr"], ["gdpr"], "2025-12-31", 20, 6, True
    )

    assert a == b
    assert a != retrieval_key(
        "what is the period?", ["retention_policy", "gdpr"], ["gdpr"], "2025-12-31", 20, 6, False
    )


def test_retrieval_cache_hands_out_copies_so_later_mutation_does_not_leak():
    cache = RetrievalCache(max_entries=5, ttl_seconds=60)
    key = retrieval_key("q", ["retention_policy"], None, "2025-12-31", 20, 6, True)
    original = [_chunk("4.2")]

    cache.put(key, (original, True, [], []), elapsed_ms=120.0)
    first = cache.get(key)
    first[0][0].rerank_score = 0.99
    second = cache.get(key)

    assert second[0][0].rerank_score is None
    assert original[0].rerank_score is None
    assert cache.saved_ms == 240.0
    assert cache.report()["hits"] == 2


def test_retrieval_cache_does_not_store_empty_results():
    cache = RetrievalCache(max_entries=5, ttl_seconds=60)
    key = retrieval_key("q", ["retention_policy"], None, "2025-12-31", 20, 6, True)

    cache.put(key, ([], False, [], []), elapsed_ms=5.0)

    assert cache.get(key) is None
    assert len(cache) == 0


def test_retrieve_policy_evidence_uses_the_cache_on_the_second_call(monkeypatch):
    from src.retrieval import hybrid_search

    monkeypatch.setattr(settings, "enable_retrieval_cache", True)
    monkeypatch.setattr(hybrid_search, "retrieval_cache", RetrievalCache(max_entries=5, ttl_seconds=60))

    calls = {"count": 0}

    class _Retriever:
        def retrieve(self, bundle):
            calls["count"] += 1

            return []

    monkeypatch.setattr(hybrid_search, "build_fusion_retriever", lambda *a, **k: (_Retriever(), True))
    monkeypatch.setattr(
        hybrid_search, "to_retrieved_chunks", lambda nodes, fused, reranked=False: [_chunk("4.2")]
    )

    first = hybrid_search.retrieve_policy_evidence("q", ["retention_policy"], use_rerank=False)
    second = hybrid_search.retrieve_policy_evidence("q", ["retention_policy"], use_rerank=False)

    assert calls["count"] == 1
    assert [c.chunk_id for c in first[0]] == [c.chunk_id for c in second[0]]
    assert second[2] == ["reranker"]


@pytest.fixture
def llm_cache():
    previous = get_llm_cache()
    cache = TTLLLMCache(max_entries=10, ttl_seconds=60)
    set_llm_cache(cache)

    yield cache

    set_llm_cache(previous)


def test_llm_cache_serves_the_second_identical_call_from_memory(llm_cache):
    model = GenericFakeChatModel(messages=iter([AIMessage(content="first"), AIMessage(content="second")]))

    first = model.invoke([HumanMessage(content="same prompt")])
    second = model.invoke([HumanMessage(content="same prompt")])
    third = model.invoke([HumanMessage(content="a different prompt")])

    assert first.content == "first"
    assert second.content == "first"
    assert third.content == "second"
    assert llm_cache.report()["hits"] == 1
    assert llm_cache.report()["misses"] == 2


def test_llm_cache_is_truthy_even_when_empty(llm_cache):
    assert bool(llm_cache) is True
    assert llm_cache.size() == 0


def test_llm_cache_attributes_hits_to_the_request_in_context(llm_cache):
    from src.llm_routing import ledger as ledger_module
    from src.llm_routing.ledger import CostLedger
    from src.observability.logging_config import request_context

    fresh = CostLedger()
    original = ledger_module.cost_ledger
    ledger_module.cost_ledger = fresh

    try:
        model = GenericFakeChatModel(messages=iter([AIMessage(content="x")]))

        with request_context(request_id="req-cache"):
            model.invoke([HumanMessage(content="p")])
            model.invoke([HumanMessage(content="p")])

        assert fresh.summary("req-cache")["cache_hits"] == 1
    finally:
        ledger_module.cost_ledger = original


def test_model_name_is_read_from_the_llm_string():
    azure = '{"kwargs": {"deployment_name": "gpt-4o-mini", "temperature": 0.0}}---[(\'stop\', None)]'
    gateway = '{"kwargs": {"model_name": "complex-agent"}}---[]'

    assert _model_from(azure) == "gpt-4o-mini"
    assert _model_from(gateway) == "complex-agent"
    assert _model_from("nothing here") == "unknown"


def test_cache_report_and_invalidation_cover_every_cache(monkeypatch):
    monkeypatch.setattr(settings, "enable_semantic_cache", False)
    from src.cache import response_cache as response_cache_module

    fresh = ResponseCache(max_entries=5, ttl_seconds=60, semantic_threshold=1.0, embedder=None, clock=Clock())
    monkeypatch.setattr(response_cache_module, "response_cache", fresh)
    monkeypatch.setattr(registry, "retrieval_cache", RetrievalCache(max_entries=5, ttl_seconds=60))

    fresh.put("s", "q", _answer(), 200, source_request_id="r", produced_in_ms=10.0, tokens_spent=5)

    report = registry.cache_report()

    assert set(report["enabled"]) == {"response", "semantic", "retrieval", "llm"}
    assert report["response"]["entries"] == 1
    assert report["totals"]["hits"] == 0

    cleared = registry.invalidate_corpus_caches("test")

    assert cleared["response"] == 1
    assert cleared["retrieval"] == 0
    assert math.isclose(registry.cache_report()["response"]["entries"], 0)
