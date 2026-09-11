"""The real graph, the real nodes, a fake chat model: routing decisions end to end.

Unlike ``test_graph_smoke.py`` (which replaces the LLM-backed nodes), this
runs the genuine ``intent_classification``, ``risk_assessment``, ``planner``,
``rag_path``, ``compliance_validation`` and ``confidence_scoring`` nodes and
only fakes the chat model underneath them. It proves that

* ``routed_model`` picks the small tier for a simple question and the strong
  tier for a complex one, inside a real LangGraph run;
* every routable call leaves a ``model_routing`` entry in the final state,
  with the reason the trace and the reviewer package will show;
* the LLM logging callback and the cost ledger see the calls without
  breaking anything.
"""

import time

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from configs import llms
from configs.settings import settings
from src.llm_routing import router as router_module
from src.llm_routing.router import RoutingStats
from src.nodes import rag_path as rag_path_module
from src.nodes.risk_assessment import RiskClassification
from src.nodes.validation import ValidatorOutput
from src.schemas.enums import EvidencePath, Intent, RiskLevel, TerminalOutcome
from src.schemas.models import (
    DraftAnswer,
    EvidencePlan,
    IntentResult,
    PlanStep,
    RetrievedChunk,
)
from tests.conftest import base_state

CLAUSE = "Retention Policy §4.2"
ANSWER = f"Customer invoices are retained for seven years after the transaction date [{CLAUSE}]."


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="retention-4.2",
        doc_type="retention_policy",
        document_title="Retention Policy",
        section="Financial records",
        clause_number="4.2",
        version="2.0",
        content="Customer invoices are retained for seven years after the transaction date.",
        rerank_score=0.8,
    )


class FakeChatModel:
    """Answers every structured-output request with a canned, schema-valid object."""

    def __init__(self, deployment: str, log: list[tuple[str, str]]):
        self.deployment = deployment
        self.log = log

    def with_structured_output(self, schema: type[BaseModel]):
        canned = {
            IntentResult: IntentResult(intent=Intent.POLICY_LOOKUP),
            RiskClassification: RiskClassification(level=RiskLevel.LOW),
            EvidencePlan: EvidencePlan(
                path=EvidencePath.RAG,
                steps=[PlanStep(order=1, source="policy_kb", objective="find the period", must_prove="7y")],
            ),
            DraftAnswer: DraftAnswer(answer=ANSWER, cited_clauses=[CLAUSE]),
            ValidatorOutput: ValidatorOutput(defects=[], grounded_claim_ratio=1.0),
        }

        if schema not in canned:  # pragma: no cover - guards against a new node type
            raise AssertionError(f"no canned answer for {schema.__name__}")

        def _answer(_input, config=None):
            self.log.append((schema.__name__, self.deployment))

            return canned[schema]

        return RunnableLambda(_answer)


@pytest.fixture
def graph(memory_graph, monkeypatch):
    calls: list[tuple[str, str]] = []

    # ``_build_chat_model`` takes a ``ProviderSpec``; the deployment name is ``spec.model``.
    monkeypatch.setattr(llms, "_build_chat_model", lambda spec, temperature: FakeChatModel(spec.model, calls))
    monkeypatch.setattr(rag_path_module, "retrieve_policy_evidence", lambda **kwargs: ([_chunk()], True, []))
    monkeypatch.setattr(router_module, "routing_stats", RoutingStats())
    monkeypatch.setattr(settings, "model_routing_strategy", "heuristic")
    monkeypatch.setattr(settings, "use_model_routing_yaml", False)
    monkeypatch.setattr(settings, "llm_gateway_url", "")

    compiled = memory_graph.get_compiled_graph()

    def _run(query: str, thread_id: str) -> dict:
        state = base_state(
            query, thread_id=thread_id, request_id=f"req-{thread_id}", deadline_ts=time.monotonic() + 600
        )

        return compiled.invoke(state, config={"configurable": {"thread_id": thread_id}, "recursion_limit": 40})

    _run.calls = calls

    return _run


def test_a_simple_question_is_generated_and_validated_on_the_small_tier(graph):
    final = graph("What is the retention period for customer invoices?", "simple")

    assert final["terminal_outcome"] == TerminalOutcome.ANSWERED.value

    routing = {entry["node"]: entry for entry in final["model_routing"]}

    assert set(routing) == {"rag_generate", "compliance_validation"}
    assert routing["rag_generate"]["tier"] == "small"
    assert routing["compliance_validation"]["tier"] == "small"
    assert routing["rag_generate"]["complexity"]["label"] == "simple"
    assert routing["rag_generate"]["strategy"] == "heuristic"

    small = settings.azure_openai_small_deployment
    assert ("DraftAnswer", small) in graph.calls
    assert ("ValidatorOutput", small) in graph.calls
    assert ("IntentResult", small) in graph.calls
    assert ("RiskClassification", small) in graph.calls
    # a policy lookup can only take the RAG path, so the planner model is not asked
    assert not any(schema == "EvidencePlan" for schema, _ in graph.calls)
    assert final["path_decision"]["planner"] == "deterministic"


def test_a_complex_question_is_generated_and_validated_on_the_strong_tier(graph):
    final = graph("Why does the legal hold override the retention period for invoices?", "complex")

    assert final["terminal_outcome"] == TerminalOutcome.ANSWERED.value

    routing = {entry["node"]: entry for entry in final["model_routing"]}

    assert routing["rag_generate"]["tier"] == "strong"
    assert routing["compliance_validation"]["tier"] == "strong"
    assert "legal hold" in routing["rag_generate"]["reason"] or "override" in routing["rag_generate"]["reason"]

    strong = settings.azure_openai_strong_deployment
    assert ("DraftAnswer", strong) in graph.calls
    assert ("ValidatorOutput", strong) in graph.calls


def test_static_strategy_keeps_the_previous_behaviour(graph, monkeypatch):
    monkeypatch.setattr(settings, "model_routing_strategy", "static")

    final = graph("What is the retention period for customer invoices?", "static")

    routing = {entry["node"]: entry for entry in final["model_routing"]}

    assert routing["rag_generate"]["tier"] == "strong"
    assert routing["rag_generate"]["strategy"] == "static"


def test_routing_decisions_reach_the_trace_spans_and_the_stats(graph):
    final = graph("What is the retention period for customer invoices?", "trace")

    completed = [span["node"] for span in final["trace"] if span.get("status") == "completed"]

    assert "rag_path" in completed and "compliance_validation" in completed

    report = router_module.routing_report()

    assert report["routable_decisions"] == 2
    assert report["routable_small_share"] == 1.0
