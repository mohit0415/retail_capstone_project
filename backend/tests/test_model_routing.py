"""Cost & latency: complexity assessment, tier routing, pricing and the ledger."""

import time

import pytest

from configs import llms
from configs.settings import settings
from src.llm_routing import ledger as ledger_module
from src.llm_routing import pricing
from src.llm_routing.complexity import assess_complexity
from src.llm_routing.ledger import CostLedger, tier_of
from src.llm_routing.router import (
    FIXED_SMALL,
    FIXED_STRONG,
    ROUTABLE,
    RoutingStats,
    route_tier,
    routing_report,
)
from src.schemas.enums import Intent, RiskLevel
from src.schemas.models import IntentResult, RiskAssessment


def test_a_short_definition_question_is_simple():
    result = assess_complexity("What is the retention period for customer invoices?")

    assert result.label == "simple"
    assert result.tier == "small"
    assert "retention period" in result.matched_simple
    assert result.matched_complex == []


def test_a_complex_trigger_phrase_wins_outright():
    result = assess_complexity("Can we transfer customer data to a vendor in a restricted jurisdiction?")

    assert result.complex
    assert result.tier == "strong"
    assert "jurisdiction" in result.matched_complex


def test_long_questions_are_complex(monkeypatch):
    monkeypatch.setattr(settings, "routing_complex_word_count", 10)

    result = assess_complexity("the policy says that vendors must be reviewed every year " * 3)

    assert result.complex
    assert any("exceeds 10" in reason for reason in result.reasons)


def test_multi_part_questions_are_complex():
    result = assess_complexity("Is the vendor approved? And also is the approval still valid?")

    assert result.complex
    assert "multi-part question" in result.reasons


@pytest.mark.parametrize("risk", ["High", "high"])
def test_high_risk_is_always_complex(risk):
    result = assess_complexity("what is the retention period", risk_level=risk)

    assert result.complex
    assert "risk level is High" in result.reasons


@pytest.mark.parametrize("intent", ["compliance_check", "incident_guidance"])
def test_rule_against_record_intents_are_complex(intent):
    result = assess_complexity("what is the retention period", intent=intent)

    assert result.complex


@pytest.mark.parametrize("path", ["hybrid", "agentic", "high_risk_panel"])
def test_two_source_evidence_paths_are_complex(path):
    assert assess_complexity("what is the retention period", evidence_path=path).complex


def test_rag_path_with_a_plain_lookup_stays_simple():
    result = assess_complexity(
        "what is the retention period", intent="policy_lookup", evidence_path="rag", risk_level="Low"
    )

    assert not result.complex
    assert 0.0 <= result.score <= 1.0


def test_the_score_grows_with_the_number_of_signals():
    weak = assess_complexity("what is the retention period", risk_level="Medium")
    strong = assess_complexity(
        "why does the legal hold override the retention period?", risk_level="High", intent="compliance_check"
    )

    assert strong.score > weak.score
    assert strong.as_dict()["label"] == "complex"


def _state(
    query: str,
    risk: RiskLevel = RiskLevel.LOW,
    intent: Intent = Intent.POLICY_LOOKUP,
    seconds_left: float = 30.0,
    tokens_left: int = 40000,
) -> dict:
    now = time.monotonic()

    return {
        "request_id": "req-route",
        "standalone_query": query,
        "risk": RiskAssessment(final_level=risk),
        "intent": IntentResult(intent=intent),
        "resolved_entities": [],
        "conversation_history": [],
        "routed_path": "rag",
        "started_ts": now,
        "deadline_ts": now + seconds_left,
        "token_budget": tokens_left,
        "tokens_spent": 0,
    }


@pytest.fixture(autouse=True)
def _fresh_routing_stats(monkeypatch):
    from src.llm_routing import router

    monkeypatch.setattr(router, "routing_stats", RoutingStats())


def test_node_classes_do_not_overlap():
    assert not (FIXED_SMALL & FIXED_STRONG)
    assert not (FIXED_SMALL & ROUTABLE)
    assert not (FIXED_STRONG & ROUTABLE)


@pytest.mark.parametrize("node", sorted(FIXED_STRONG))
def test_panel_agents_are_always_strong_whatever_the_strategy(node):
    for strategy in ("static", "heuristic", "cost_saver", "quality_first"):
        decision = route_tier(
            node, _state("what is the retention period", seconds_left=0.5), strategy=strategy
        )

        assert decision.tier == "strong"
        assert decision.routable is False


@pytest.mark.parametrize("node", ["intent_classification", "planner", "sql_narration", "reflection"])
def test_classification_nodes_are_always_small(node):
    assert (
        route_tier(node, _state("why does the legal hold override retention?"), strategy="quality_first").tier
        == "small"
    )


def test_heuristic_routes_a_simple_question_to_the_small_tier():
    decision = route_tier(
        "rag_generate", _state("What is the retention period for customer invoices?"), strategy="heuristic"
    )

    assert decision.routable
    assert decision.tier == "small"
    assert decision.assessment is not None and decision.assessment.label == "simple"
    assert decision.deployment == settings.azure_openai_small_deployment
    assert decision.as_dict()["complexity"]["label"] == "simple"


def test_heuristic_routes_a_complex_question_to_the_strong_tier():
    decision = route_tier(
        "rag_generate", _state("Does the legal hold override the deletion request?"), strategy="heuristic"
    )

    assert decision.tier == "strong"
    assert decision.deployment == settings.azure_openai_strong_deployment


def test_static_keeps_the_fixed_strong_tier_for_routable_nodes():
    assert (
        route_tier("rag_generate", _state("What is the retention period?"), strategy="static").tier
        == "strong"
    )


def test_cost_saver_uses_small_unless_risk_is_high():
    assert (
        route_tier(
            "hybrid_generate",
            _state("Does the legal hold override the deletion request?"),
            strategy="cost_saver",
        ).tier
        == "small"
    )
    assert (
        route_tier(
            "hybrid_generate",
            _state("what is the retention period", risk=RiskLevel.HIGH),
            strategy="cost_saver",
        ).tier
        == "strong"
    )


def test_quality_first_is_never_downgraded_by_budget_pressure():
    decision = route_tier(
        "rag_generate",
        _state("what is the retention period", seconds_left=1.0, tokens_left=500),
        strategy="quality_first",
    )

    assert decision.tier == "strong"
    assert decision.budget_pressure is True
    assert decision.downgraded is False


def test_budget_pressure_downgrades_a_strong_choice_to_small(monkeypatch):
    monkeypatch.setattr(settings, "routing_latency_pressure_seconds", 8.0)

    decision = route_tier(
        "rag_generate",
        _state("Does the legal hold override the deletion request?", seconds_left=3.0),
        strategy="heuristic",
    )

    assert decision.tier == "small"
    assert decision.downgraded is True
    assert "budget pressure" in decision.reason
    assert decision.seconds_remaining is not None and decision.seconds_remaining < 8.0


def test_token_pressure_downgrades_too(monkeypatch):
    monkeypatch.setattr(settings, "routing_token_pressure", 6000)

    decision = route_tier(
        "compliance_validation",
        _state("Does the legal hold override the deletion request?", tokens_left=1000),
        strategy="heuristic",
    )

    assert decision.tier == "small"
    assert decision.downgraded


def test_high_risk_is_exempt_from_budget_pressure():
    decision = route_tier(
        "compliance_validation",
        _state("what is the retention period", risk=RiskLevel.HIGH, seconds_left=1.0),
        strategy="heuristic",
    )

    assert decision.tier == "strong"
    assert decision.downgraded is False


def test_the_configured_strategy_is_the_default(monkeypatch):
    monkeypatch.setattr(settings, "model_routing_strategy", "cost_saver")

    assert (
        route_tier("rag_generate", _state("Does the legal hold override the deletion request?")).strategy
        == "cost_saver"
    )


def test_routing_without_a_state_still_works_on_a_bare_query():
    decision = route_tier("rag_generate", None, strategy="heuristic", query="what is the retention period")

    assert decision.tier == "small"
    assert decision.seconds_remaining is None


def test_routing_report_counts_decisions():
    route_tier("rag_generate", _state("what is the retention period"), strategy="heuristic")
    route_tier(
        "rag_generate", _state("Does the legal hold override the deletion request?"), strategy="heuristic"
    )
    route_tier("planner", _state("x"), strategy="heuristic")

    report = routing_report()

    assert report["routable_decisions"] == 2
    assert report["routable_small_share"] == 0.5
    assert report["decisions_by_node"]["rag_generate"] == {"small": 1, "strong": 1}
    assert "tiers" in report and set(report["tiers"]) == {"small", "strong"}


def test_routed_model_returns_a_model_for_the_chosen_tier(monkeypatch):
    built = []

    def _fake_build(spec, temperature):
        # ``_build_chat_model`` receives a ``ProviderSpec``; the deployment is ``spec.model``.
        built.append((spec.model, temperature))

        return f"model:{spec.model}"

    monkeypatch.setattr(llms, "_build_chat_model", _fake_build)
    monkeypatch.setattr(settings, "model_routing_strategy", "heuristic")
    monkeypatch.setattr(settings, "use_model_routing_yaml", False)
    monkeypatch.setattr(settings, "llm_gateway_url", "")

    model, decision = llms.routed_model("rag_generate", _state("what is the retention period"))

    assert decision.tier == "small"
    assert model == f"model:{settings.azure_openai_small_deployment}"
    assert built == [(settings.azure_openai_small_deployment, 0.1)]


def test_model_for_is_unchanged_and_uses_the_fixed_table(monkeypatch):
    monkeypatch.setattr(llms, "_build_chat_model", lambda spec, temperature: (spec.model, temperature))
    monkeypatch.setattr(settings, "use_model_routing_yaml", False)
    monkeypatch.setattr(settings, "llm_gateway_url", "")

    assert llms.model_for("rag_generate") == (settings.azure_openai_strong_deployment, 0.1)
    assert llms.model_for("intent_classification") == (settings.azure_openai_small_deployment, 0.0)
    assert llms.model_for("panel_challenger") == (settings.azure_openai_strong_deployment, 0.4)


def test_gateway_aliases_replace_deployment_names(monkeypatch):
    monkeypatch.setattr(settings, "llm_gateway_url", "http://localhost:4000/v1")

    assert llms.deployment_for("small") == settings.llm_gateway_small_model
    assert llms.deployment_for("strong") == settings.llm_gateway_strong_model


def test_only_deterministic_calls_use_the_llm_cache(monkeypatch):
    monkeypatch.setattr(settings, "enable_llm_cache", True)

    assert llms._cache_flag(0.0) is None
    assert llms._cache_flag(0.4) is False

    monkeypatch.setattr(settings, "enable_llm_cache", False)

    assert llms._cache_flag(0.0) is False


@pytest.fixture(autouse=True)
def _reset_prices():
    pricing.reset_price_table()

    yield

    pricing.reset_price_table()


def test_versioned_deployment_names_resolve_by_longest_prefix():
    assert pricing.price_for("gpt-4o-mini-2024-07-18").model == "gpt-4o-mini"
    assert pricing.price_for("gpt-4o-2024-08-06").model == "gpt-4o"
    assert pricing.price_for("azure/gpt-4o").model == "gpt-4o"


def test_unknown_models_cost_nothing_but_do_not_fail():
    price = pricing.price_for("mystery-model")

    assert price.input_per_million == 0.0
    assert pricing.estimate_cost("mystery-model", 1000, 1000) == 0.0


def test_cost_is_priced_per_million_tokens():
    usd = pricing.estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)

    assert usd == pytest.approx(0.15 + 0.60)


def test_model_prices_json_overrides_the_table(monkeypatch):
    monkeypatch.setattr(
        settings,
        "model_prices_json",
        '{"gpt-4o": {"input": 1.0, "output": 2.0}, "my-local": {"input": 0, "output": 0}}',
    )
    pricing.reset_price_table()

    assert pricing.price_for("gpt-4o").input_per_million == 1.0
    assert pricing.price_for("my-local").output_per_million == 0.0


def test_malformed_model_prices_json_falls_back_to_defaults(monkeypatch):
    monkeypatch.setattr(settings, "model_prices_json", "{not json")
    pricing.reset_price_table()

    assert pricing.price_for("gpt-4o-mini").input_per_million == 0.15


def test_ledger_accumulates_per_request_and_process_wide():
    ledger = CostLedger()

    ledger.record_call(
        "r1", node="rag_generate", model="gpt-4o", prompt_tokens=1000, completion_tokens=500, elapsed_ms=800.0
    )
    ledger.record_call(
        "r1",
        node="intent_classification",
        model="gpt-4o-mini",
        prompt_tokens=200,
        completion_tokens=50,
        elapsed_ms=100.0,
    )
    ledger.record_call(
        "r2", node="rag_generate", model="gpt-4o-mini", prompt_tokens=1000, completion_tokens=500
    )
    ledger.record_cache_hit("r2", model="gpt-4o-mini")

    r1 = ledger.summary("r1")
    assert r1["calls"] == 2
    assert r1["total_tokens"] == 1750
    assert r1["usd"] == pytest.approx((1000 * 2.5 + 500 * 10.0) / 1e6 + (200 * 0.15 + 50 * 0.6) / 1e6)
    assert set(r1["by_node"]) == {"rag_generate", "intent_classification"}

    r2 = ledger.summary("r2")
    assert r2["cache_hits"] == 1

    report = ledger.report()
    assert report["process"]["calls"] == 3
    assert report["requests_tracked"] == 2
    assert report["by_tier"]["strong"]["calls"] == 1
    assert report["by_tier"]["small"]["calls"] == 2
    assert report["small_tier_share"] == pytest.approx(2 / 3, abs=1e-3)
    assert report["usd_per_request_peak"] >= report["usd_per_request_mean"]


def test_ledger_summary_for_an_unknown_request_is_empty():
    assert CostLedger().summary("nope")["calls"] == 0


def test_ledger_flags_a_request_over_the_soft_budget(monkeypatch, caplog):
    monkeypatch.setattr(settings, "cost_budget_usd_per_request", 0.001)
    ledger = CostLedger()

    with caplog.at_level("WARNING", logger="src.llm_routing.ledger"):
        ledger.record_call(
            "r-big", node="rag_generate", model="gpt-4o", prompt_tokens=100_000, completion_tokens=10_000
        )

    assert ledger.summary("r-big")["over_budget"] is True
    assert "cost budget exceeded" in caplog.text
    assert ledger.report()["requests_over_budget"] == 1


def test_ledger_bounds_the_number_of_tracked_requests():
    ledger = CostLedger(max_requests=2)

    for index in range(5):
        ledger.record_call(f"r{index}", node="n", model="gpt-4o-mini", prompt_tokens=1, completion_tokens=1)

    assert ledger.report()["requests_tracked"] == 2
    assert ledger.report()["process"]["calls"] == 5


def test_tier_of_recognises_deployments_and_gateway_aliases(monkeypatch):
    # Pin distinct deployment names: a .env that serves both tiers from the same
    # deployment (small == strong) makes a name-based tier lookup ambiguous.
    monkeypatch.setattr(settings, "azure_openai_small_deployment", "gpt-4o-mini")
    monkeypatch.setattr(settings, "azure_openai_strong_deployment", "gpt-4o")

    assert tier_of(settings.azure_openai_small_deployment) == "small"
    assert tier_of(settings.azure_openai_strong_deployment + "-2024-08-06") == "strong"
    assert tier_of(settings.llm_gateway_strong_model) == "strong"
    assert tier_of("unknown-thing") == "other"


def test_the_module_ledger_is_the_one_the_callback_writes_to():
    assert isinstance(ledger_module.cost_ledger, CostLedger)
