"""Latency and answer-quality fixes (Sep 2026).

Each test pins one behaviour that was read out of the production log:

* per-request checkpoints, so a follow-up never inherits tokens, budget stops or chunks
* risk probes that only count rows for the vendor / department the question names
* the intent and risk classifiers running side by side
* no planner / rewrite / validator model call when it cannot change the outcome
* a RAG repair that actually changes something, capped at one attempt
* no automatic re-plan for High risk or for a no-match retrieval
* the honest "I don't know" answer when the documents do not contain the answer
* the agent workflow summary the chat UI draws
"""

import time
from contextlib import contextmanager

import pytest

from src.graph import routing
from src.schemas.enums import DefectType, EvidencePath, Intent, RiskLevel, Role, TerminalOutcome
from src.schemas.models import (
    ConfidenceBreakdown,
    Defect,
    DraftAnswer,
    EvidencePlan,
    IntentResult,
    PlanStep,
    ResolvedEntity,
    RetrievedChunk,
    RiskAssessment,
    ValidationReport,
)
from tests.conftest import base_state
from tests.test_graph_smoke import Stubs


def _chunk(clause: str = "4", score: float | None = 0.9, title: str = "Data Retention And Archival Policy") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"retention-{clause}-{score}",
        doc_type="retention_policy",
        document_title=title,
        section="Retention Period Standards",
        clause_number=clause,
        version="1.0",
        content="Customer transaction records: 7 years from transaction closure.",
        rerank_score=score,
    )


def _failed(defect: DefectType = DefectType.MISSING_CITATION) -> ValidationReport:
    return ValidationReport(passed=False, defects=[Defect(defect_type=defect, description="d")])


def _routing_state(**overrides) -> dict:
    state = {
        "deadline_ts": time.monotonic() + 60,
        "token_budget": 35000,
        "tokens_spent": 0,
        "reflection_count": 0,
        "retrieved_chunks": [_chunk()],
        "sql_evidence": None,
    }
    state.update(overrides)

    return state


# ---------------------------------------------------------------------------
# per-request checkpoint
# ---------------------------------------------------------------------------


def test_every_request_gets_its_own_checkpoint_key():
    from src.api.routes import checkpoint_thread_id

    assert checkpoint_thread_id("chat-1", "req-a") == "chat-1:req-a"
    assert checkpoint_thread_id("chat-1", "req-a") != checkpoint_thread_id("chat-1", "req-b")


def test_a_second_request_in_the_same_chat_starts_from_a_clean_state(memory_graph, monkeypatch):
    from src.api.routes import checkpoint_thread_id

    Stubs().install(memory_graph, monkeypatch)
    graph = memory_graph.get_compiled_graph()

    def _run(request_id: str) -> dict:
        state = base_state(thread_id="chat-1", request_id=request_id)
        config = {"configurable": {"thread_id": checkpoint_thread_id("chat-1", request_id)}, "recursion_limit": 40}

        return graph.invoke(state, config=config)

    first = _run("req-1")
    second = _run("req-2")

    assert first["tokens_spent"] == second["tokens_spent"] == 1200
    assert len(second["retrieved_chunks"]) == 1
    assert [span["node"] for span in second["trace"]].count("input_guardrail") == 1


def test_the_reviewer_resume_finds_the_per_request_checkpoint_first_and_falls_back_to_the_chat_key():
    from src.api.routes import _review_snapshot

    class _Snapshot:
        def __init__(self, values):
            self.values = values

    class _Graph:
        def __init__(self, stored):
            self.stored = stored
            self.asked = []

        def get_state(self, config):
            key = config["configurable"]["thread_id"]
            self.asked.append(key)

            return _Snapshot(self.stored.get(key, {}))

    new_style = _Graph({"chat-1:req-1": {"draft": "x"}})
    config, snapshot = _review_snapshot(new_style, {"configurable": {"thread_id": "chat-1"}}, "chat-1", "req-1")
    assert config["configurable"]["thread_id"] == "chat-1:req-1"
    assert snapshot.values == {"draft": "x"}

    legacy = _Graph({"chat-1": {"draft": "old"}})
    config, snapshot = _review_snapshot(legacy, {"configurable": {"thread_id": "chat-1"}}, "chat-1", "req-9")
    assert legacy.asked == ["chat-1:req-9", "chat-1"]
    assert config["configurable"]["thread_id"] == "chat-1"


# ---------------------------------------------------------------------------
# risk probes
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_probe_db(monkeypatch):
    from src.nodes import risk_assessment

    executed: list[tuple[str, dict]] = []

    class _Result:
        def fetchone(self):
            return {"hits": 3}

    class _Conn:
        def execute(self, statement, params):
            executed.append((statement, params))

            return _Result()

    @contextmanager
    def _connection():
        yield _Conn()

    monkeypatch.setattr(risk_assessment, "read_only_connection", _connection)

    return executed


def test_a_generic_retention_question_runs_no_probe(fake_probe_db):
    from src.nodes.risk_assessment import run_probes

    signals, ran, skipped = run_probes("data_retention", vendor_id=None, department=None)

    assert signals == []
    assert ran == []
    assert skipped == ["retention_review_overdue", "retention_pending_approval"]
    assert fake_probe_db == []


def test_an_overdue_retention_review_for_a_named_department_is_medium_not_high(fake_probe_db):
    from src.nodes.risk_assessment import run_probes

    signals, ran, _ = run_probes("data_retention", vendor_id=None, department="Finance")

    assert ran == ["retention_review_overdue", "retention_pending_approval"]
    assert {signal.level for signal in signals} == {RiskLevel.MEDIUM}
    assert all(params["department"] == "Finance" for _, params in fake_probe_db)


def test_a_legal_hold_on_a_named_vendor_is_still_high(fake_probe_db):
    from src.nodes.risk_assessment import run_probes

    signals, ran, _ = run_probes("data_erasure", vendor_id=7, department=None)

    assert "legal_hold_present" in ran
    assert RiskLevel.HIGH in {signal.level for signal in signals}


def test_vendor_only_probes_never_run_on_a_department_anchor(fake_probe_db):
    from src.nodes.risk_assessment import run_probes

    _, ran, skipped = run_probes("vendor_engagement", vendor_id=None, department="Finance")

    assert ran == []
    assert len(skipped) == 3


def test_the_risk_node_reuses_the_classification_started_next_to_intent(monkeypatch, no_db, no_tracing):
    from src.nodes import risk_assessment

    monkeypatch.setattr(
        risk_assessment, "classify_risk_l2", lambda *a, **k: pytest.fail("the L2 classifier ran twice")
    )

    state = base_state(query="What is the retention period for customer transaction records?")
    state["risk_l2"] = {
        "query": state["standalone_query"],
        "level": "Low",
        "scenario_id": "data_retention",
        "rationale": "standard retention period",
    }

    result = risk_assessment.risk_assessment_node(state)

    assert result["risk"].final_level is RiskLevel.MEDIUM  # lexical floor for "retention period"
    assert result["risk"].probes_skipped == ["retention_review_overdue", "retention_pending_approval"]
    assert not result["risk"].disagreement


def test_a_stale_precomputed_classification_is_not_trusted(monkeypatch, no_db, no_tracing):
    from src.nodes import risk_assessment

    calls = []

    def _classify(state, query):
        calls.append(query)

        return risk_assessment.RiskClassification(level=RiskLevel.LOW)

    monkeypatch.setattr(risk_assessment, "classify_risk_l2", _classify)

    state = base_state(query="Can I accept a gift from a supplier during a tender?")
    state["risk_l2"] = {"query": "a different question", "level": "High", "scenario_id": "", "rationale": ""}

    result = risk_assessment.risk_assessment_node(state)

    assert calls == [state["standalone_query"]]
    assert result["risk"].final_level is RiskLevel.LOW


def test_intent_and_risk_are_classified_side_by_side(monkeypatch, no_db, no_tracing):
    from src.nodes import intent_classification

    monkeypatch.setattr(intent_classification, "_classify_intent", lambda state, query: IntentResult(intent=Intent.POLICY_LOOKUP))
    monkeypatch.setattr(
        intent_classification,
        "_classify_risk",
        lambda state, query: {"query": query, "level": "Low", "scenario_id": "", "rationale": ""},
    )

    result = intent_classification.intent_classification_node(base_state())

    assert result["intent"].intent is Intent.POLICY_LOOKUP
    assert result["risk_l2"]["level"] == "Low"


# ---------------------------------------------------------------------------
# repair routing
# ---------------------------------------------------------------------------


def test_a_failed_rag_answer_gets_one_repair_not_two():
    assert routing.after_validation(_routing_state(validation=_failed(), evidence_path="rag")) == "reflect"
    assert (
        routing.after_validation(_routing_state(validation=_failed(), evidence_path="rag", reflection_count=1))
        == "escalate"
    )


def test_a_hybrid_answer_keeps_the_configured_repair_count():
    state = _routing_state(validation=_failed(), evidence_path="hybrid", reflection_count=1)

    assert routing.after_validation(state) == "reflect"


def test_a_failed_high_risk_answer_goes_to_the_reviewer_without_re_running_the_panel():
    state = _routing_state(
        validation=_failed(),
        evidence_path="high_risk_panel",
        risk=RiskAssessment(final_level=RiskLevel.HIGH),
    )

    assert routing.after_validation(state) == "escalate"


def test_a_no_match_retrieval_is_never_repaired_and_gets_the_honest_no_answer():
    state = _routing_state(validation=_failed(), evidence_path="hybrid", retrieved_chunks=[_chunk(score=0.0085)])

    assert routing.after_validation(state) == "not_found"


def test_a_no_match_retrieval_still_escalates_when_the_user_asked_for_a_human():
    state = _routing_state(
        validation=_failed(),
        evidence_path="hybrid",
        retrieved_chunks=[_chunk(score=0.0085)],
        raw_query="retention for invoices? I need a legal review",
    )

    assert routing.after_validation(state) == "escalate"


def test_a_repair_that_cannot_fit_in_the_token_budget_is_not_started():
    state = _routing_state(validation=_failed(), evidence_path="rag", token_budget=10000, tokens_spent=5000)

    assert routing.repair_token_estimate(state) > 5000
    assert routing.after_validation(state) == "escalate"


def test_a_repair_that_cannot_fit_in_the_deadline_is_not_started():
    state = _routing_state(validation=_failed(), evidence_path="rag", deadline_ts=time.monotonic() + 3)

    assert routing.after_validation(state) == "escalate"


def test_the_rag_repair_makes_no_model_call_and_carries_the_defects(monkeypatch, no_db, no_tracing):
    from src.nodes import reflection

    monkeypatch.setattr(reflection, "model_for", lambda _name: pytest.fail("reflection model called for rag"))

    state = base_state()
    state.update(
        evidence_path="rag",
        plan=EvidencePlan(path=EvidencePath.RAG, steps=[PlanStep(order=1, source="policy_kb", objective="o", must_prove="p")]),
        validation=ValidationReport(
            passed=False,
            defects=[Defect(defect_type=DefectType.UNGROUNDED_CLAIM, description="cites §9.9 which was not retrieved")],
        ),
    )

    result = reflection.reflection_node(state)

    assert result["repair_strategy"] == "rag_widen_and_fix"
    assert result["reflection_count"] == 1
    assert "§9.9" in result["replan_directive"]
    assert result["plan_from_reflection"] is True


def test_the_rag_repair_pass_searches_wider_and_hands_the_defects_to_the_writer(monkeypatch, no_db, no_tracing):
    from src.nodes import rag_path

    searched = {}
    sent = {}

    def _retrieve(**kwargs):
        searched.update(kwargs)

        return [_chunk()], True, []

    class _Model:
        def with_structured_output(self, _schema):
            return self

        def invoke(self, messages, config=None):
            sent["messages"] = messages

            return DraftAnswer(answer="Seven years [Data Retention And Archival Policy §4].")

    class _Decision:
        tier = "small"

        def as_dict(self):
            return {"node": "rag_generate", "tier": "small"}

    monkeypatch.setattr(rag_path, "retrieve_policy_evidence", _retrieve)
    monkeypatch.setattr(rag_path, "routed_model", lambda node, state: (_Model(), _Decision()))

    state = base_state(role=Role.STORE_ASSOCIATE.value)
    state.update(reflection_count=1, replan_directive="Fix every defect: missing_citation", document_scope_request=["privacy_policy"])

    rag_path.rag_path_node(state)

    assert searched["doc_scope"] is None
    assert searched["top_k"] > 20
    assert any("Fix every defect" in str(message.content) for message in sent["messages"])


# ---------------------------------------------------------------------------
# validator
# ---------------------------------------------------------------------------


def test_the_validator_model_is_not_asked_to_confirm_a_draft_that_already_failed(monkeypatch, no_db, no_tracing):
    from src.nodes import validation

    class _Decision:
        tier = "small"

        def as_dict(self):
            return {"node": "compliance_validation", "tier": "small"}

    class _Model:
        def with_structured_output(self, _schema):
            raise AssertionError("validator model called for a draft that already failed")

    monkeypatch.setattr(validation, "routed_model", lambda node, state: (_Model(), _Decision()))

    state = base_state()
    state.update(
        retrieved_chunks=[_chunk("4")],
        draft=DraftAnswer(answer="Kept for ten years [Data Retention And Archival Policy §9.9]."),
    )

    result = validation.compliance_validation_node(state)

    assert result["validation"].passed is False
    assert "deterministic checks already failed" in result["validation_llm_skipped"]


def test_the_validator_reads_the_cited_extracts_first_and_only_a_few_others():
    from src.nodes.validation import VALIDATOR_MAX_EXTRACTS, validator_extracts

    chunks = [_chunk(str(n)) for n in range(1, 11)]
    state = {"retrieved_chunks": chunks, "draft": DraftAnswer(answer="See [Data Retention And Archival Policy §7].")}

    picked = validator_extracts(state)

    assert picked[0].clause_number == "7"
    assert len(picked) == VALIDATOR_MAX_EXTRACTS


# ---------------------------------------------------------------------------
# budget guard
# ---------------------------------------------------------------------------


def test_a_drafted_low_risk_answer_may_still_be_validated_just_after_the_deadline():
    from src.core.budget import BudgetGuard, finishing_grace_seconds

    state = {"draft": DraftAnswer(answer="x"), "risk": RiskAssessment(final_level=RiskLevel.LOW)}
    guard = BudgetGuard(deadline_ts=time.monotonic() - 2, token_budget=35000, tokens_spent=0)

    assert guard.check("compliance_validation", grace_seconds=finishing_grace_seconds("compliance_validation", state)).allowed


def test_no_grace_without_a_draft_or_for_high_risk():
    from src.core.budget import finishing_grace_seconds

    assert finishing_grace_seconds("compliance_validation", {}) == 0.0
    assert (
        finishing_grace_seconds(
            "compliance_validation",
            {"draft": DraftAnswer(answer="x"), "risk": RiskAssessment(final_level=RiskLevel.HIGH)},
        )
        == 0.0
    )
    assert finishing_grace_seconds("rag_path", {"draft": DraftAnswer(answer="x")}) == 0.0


def test_scoring_a_validated_answer_is_never_budget_stopped():
    from src.core.budget import BudgetGuard

    guard = BudgetGuard(deadline_ts=time.monotonic() - 30, token_budget=100, tokens_spent=100)

    assert guard.check("confidence_scoring").allowed
    assert guard.check("no_answer").allowed


# ---------------------------------------------------------------------------
# "I don't know"
# ---------------------------------------------------------------------------


def test_a_question_the_documents_do_not_answer_gets_an_honest_no_answer(memory_graph, monkeypatch):
    stubs = Stubs(chunks=[_chunk(score=0.0085)]).install(memory_graph, monkeypatch)
    graph = memory_graph.get_compiled_graph()

    state = base_state(
        query="What is the retention period for customer transaction records?",
        request_id="req-nf",
        thread_id="chat-nf",
        role=Role.STORE_ASSOCIATE.value,
        access_scopes=["doc:privacy_policy", "doc:infosec_policy"],
    )
    final = graph.invoke(state, config={"configurable": {"thread_id": "chat-nf:req-nf"}, "recursion_limit": 40})

    nodes = [span["node"] for span in final["trace"]]

    assert final["terminal_outcome"] == TerminalOutcome.ANSWERED.value
    assert final["answer_not_found"] is True
    assert final["draft"].answer.startswith("I don't know.")
    assert "InfoSec, Privacy" in final["draft"].answer
    assert "compliance_validation" not in stubs.visits
    assert nodes[-2:] == ["no_answer", "output_guardrail"]


def test_a_high_risk_no_match_still_goes_to_a_human(memory_graph, monkeypatch):
    Stubs(risk=RiskLevel.HIGH, chunks=[_chunk(score=0.001)]).install(memory_graph, monkeypatch)
    graph = memory_graph.get_compiled_graph()

    final = graph.invoke(base_state(request_id="req-h"), config={"configurable": {"thread_id": "t:req-h"}, "recursion_limit": 40})

    assert final["terminal_outcome"] == TerminalOutcome.ESCALATED.value
    assert not final.get("answer_not_found")


def test_a_weak_but_real_match_is_still_drafted_and_validated(memory_graph, monkeypatch):
    stubs = Stubs(chunks=[_chunk(score=0.42)]).install(memory_graph, monkeypatch)
    graph = memory_graph.get_compiled_graph()

    graph.invoke(base_state(request_id="req-w"), config={"configurable": {"thread_id": "t:req-w"}, "recursion_limit": 40})

    assert "compliance_validation" in stubs.visits


def test_the_no_answer_text_passes_the_output_guardrail_without_citations():
    from src.guardrails.output_guard import run_output_guardrail
    from src.nodes.no_answer import not_found_text

    text = not_found_text({"access_scopes": ["doc:privacy_policy", "doc:retention_policy"]})

    assert not run_output_guardrail(text, [], require_citations=False).enforcement_failed
    assert run_output_guardrail(text, [], require_citations=True).enforcement_failed


def test_a_not_found_answer_lists_no_citations():
    from src.api.routes import _citations_from

    state = {"answer_not_found": True, "retrieved_chunks": [_chunk(score=0.001)], "draft": DraftAnswer(answer="I don't know.")}

    assert _citations_from(state) == []


# ---------------------------------------------------------------------------
# rewrite, clauses, RBAC, BM25, workflow view
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "standalone"),
    [
        ("What is the retention period for customer transaction records?", True),
        ("Can I accept a gift from a supplier during a tender?", True),
        ("Does that apply to contractors as well?", False),
        ("What about vendor contracts?", False),
        ("what are the retention period standards mentioned", False),
    ],
)
def test_only_follow_ups_that_lean_on_earlier_turns_are_rewritten(query, standalone):
    from src.nodes.query_rewrite import reads_as_standalone

    assert reads_as_standalone(query) is standalone


def test_a_standalone_follow_up_skips_the_rewrite_model(monkeypatch, no_db, no_tracing):
    from src.nodes import query_rewrite

    monkeypatch.setattr(query_rewrite, "model_for", lambda _name: pytest.fail("rewrite model called"))

    state = base_state(query="What is the retention period for customer transaction records?")
    state["conversation_history"] = [{"role": "user", "content": "what does the bribery policy cover"}]

    result = query_rewrite.query_rewrite_node(state)

    assert result["rewrite_mode"] == "standalone"
    assert result["standalone_query"] == state["sanitised_query"]


def test_plain_text_pdf_headings_become_real_clauses():
    from src.ingestion.extractors.clause_extraction import extract_clauses

    text = (
        "Data Retention & Archival Policy\n"
        "1. Purpose\nThis policy defines the standards.\n"
        "4. Retention Period Standards\nExamples of retention standards include:\n1\n"
        "Customer transaction records: 7 years from transaction closure.\n"
        "4.1 Transaction Records\nPoint of sale records are kept seven years.\n"
    )

    sections = extract_clauses(text)

    assert [section.clause_number for section in sections] == ["1", "4", "4.1"]
    assert sections[1].heading == "Retention Period Standards"
    assert "7 years" in sections[1].body


def test_prose_with_a_single_numbered_line_is_left_alone():
    from src.ingestion.extractors.clause_extraction import extract_clauses

    sections = extract_clauses("Intro paragraph.\n2 Reasons We Exist\nMore prose follows here.")

    assert [section.clause_number for section in sections] == ["1"]


def test_a_store_associate_can_read_the_retention_policy_but_no_table():
    from src.auth.rbac import access_scopes_for, allowed_doc_types, allowed_tables

    scopes = access_scopes_for(Role.STORE_ASSOCIATE)

    assert "retention_policy" in allowed_doc_types(scopes)
    assert allowed_tables(scopes) == set()


def test_the_bm25_index_is_built_once_per_corpus(monkeypatch):
    from llama_index.core.schema import TextNode

    from src.retrieval import fusion

    fusion.clear_bm25_cache()

    corpus = [
        TextNode(id_=f"n{i}", text=f"retention clause {i}", metadata={"doc_type": "retention_policy"})
        for i in range(3)
    ]

    class _Docstore:
        def __init__(self):
            self.docs = {}

    class _Index:
        def __init__(self):
            self.docstore = shared_docstore

    shared_docstore = _Docstore()

    monkeypatch.setattr(fusion, "load_lexical_corpus", lambda: corpus)

    first = fusion._build_bm25_retriever(_Index(), ["retention_policy"], 20)
    second = fusion._build_bm25_retriever(_Index(), ["retention_policy"], 20)

    assert first is second

    newer = list(corpus)
    monkeypatch.setattr(fusion, "load_lexical_corpus", lambda: newer)

    assert fusion._build_bm25_retriever(_Index(), ["retention_policy"], 20) is not first


def test_the_workflow_view_names_the_agents_and_what_each_decided(memory_graph, monkeypatch):
    from src.observability.agent_steps import build_agent_steps

    Stubs().install(memory_graph, monkeypatch)
    graph = memory_graph.get_compiled_graph()

    final = graph.invoke(base_state(request_id="req-v"), config={"configurable": {"thread_id": "t:req-v"}, "recursion_limit": 40})
    steps = build_agent_steps(final)

    agents = [step["agent"] for step in steps]

    assert agents[:3] == ["Input Guardrail", "Query Rewrite", "Intent Classification Agent"]
    assert "Risk Assessment Agent" in agents
    assert "Retrieval Agent · RAG" in agents
    assert "Compliance Validation Agent" in agents
    assert all(step["status"] == "completed" for step in steps)
    assert next(step for step in steps if step["node"] == "confidence_scoring")["summary"] == "confidence 0.90"
    assert next(step for step in steps if step["node"] == "output_guardrail")["summary"].startswith("released")


def test_a_budget_stop_is_shown_as_such_in_the_workflow(memory_graph, monkeypatch):
    from src.observability.agent_steps import build_agent_steps

    Stubs().install(memory_graph, monkeypatch)
    graph = memory_graph.get_compiled_graph()

    state = base_state(request_id="req-b", deadline_ts=time.monotonic() - 1)
    final = graph.invoke(state, config={"configurable": {"thread_id": "t:req-b"}, "recursion_limit": 40})
    steps = build_agent_steps(final)

    assert steps[0]["status"] == "budget_stopped"
    assert "budget guard" in steps[0]["summary"]
    assert steps[-1]["agent"] == "Escalation Manager Agent"


def test_panel_members_appear_under_the_panel_step():
    from src.observability.agent_steps import build_agent_steps
    from src.schemas.models import PanelOpinion, PanelVerdict

    final = {
        "trace": [{"node": "multi_agent_panel", "status": "completed", "elapsed_ms": 12.0}],
        "panel_verdict": PanelVerdict(
            consensus="Escalate the gift approval.",
            dissent=["verifier disagrees"],
            opinions=[
                PanelOpinion(agent="policy_interpreter", position="Clause 3.1 requires approval", supporting_citations=["A §3.1"]),
                PanelOpinion(agent="data_verifier", position="No record", objections=["no row"]),
                PanelOpinion(agent="challenger", position="Threshold unclear", objections=["a", "b"]),
            ],
        ),
    }

    [step] = build_agent_steps(final)

    assert [child["agent"] for child in step["children"]] == [
        "Policy Interpreter",
        "Data Verifier",
        "Challenger",
        "Consensus",
    ]


def test_entity_anchor_helpers_read_resolved_entities():
    from src.nodes.risk_assessment import _first_department, _first_vendor_id
    from src.schemas.enums import EntityStatus

    state = {
        "resolved_entities": [
            ResolvedEntity(surface_form="finance", entity_type="department", canonical_name="Finance", status=EntityStatus.EXACT),
            ResolvedEntity(surface_form="acme", entity_type="vendor", resolved_id=4, status=EntityStatus.EXACT),
        ]
    }

    assert _first_department(state) == "Finance"
    assert _first_vendor_id(state) == 4


def test_confidence_breakdown_of_a_not_found_answer_is_low():
    from src.nodes.no_answer import no_answer_node

    state = base_state()
    state["retrieved_chunks"] = [_chunk(score=0.004)]

    result = no_answer_node.__wrapped__(state)

    assert isinstance(result["confidence"], ConfidenceBreakdown)
    assert result["confidence"].final_score < 0.05


def test_the_rag_path_does_not_write_a_draft_when_nothing_matched(monkeypatch, no_db, no_tracing):
    from src.nodes import rag_path

    monkeypatch.setattr(rag_path, "retrieve_policy_evidence", lambda **kwargs: ([_chunk(score=0.004)], True, []))
    monkeypatch.setattr(rag_path, "routed_model", lambda node, state: pytest.fail("answer model called on a no-match"))

    result = rag_path.rag_path_node(base_state(query="What is the parking reimbursement limit for store staff?"))

    assert result["draft"] is None
    assert result["tokens_spent"] == 0
    assert "no draft written" in result["trace"][-1]["summary"]
