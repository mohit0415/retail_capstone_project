"""Why validated answers still escalated (log of 2026-09-11, 03:39-04:20).

* every policy answer that passed validation on its rewrite then escalated on confidence: the
  retrieval signal was the raw cross-encoder score (0.11, 0.18) weighted 0.30 of the gate
* the validator model objected to correct records answers without naming a figure
* the agentic path crashed with "no running event loop" before its first tool call
* two questions were refused as "no policy or compliance subject" on a word list
"""

import asyncio
from datetime import date

import pytest
from langchain_core.runnables import RunnableLambda

from src.guardrails.scope import is_in_domain
from src.nodes import agentic_rag, validation
from src.nodes.validation import ValidatorOutput
from src.schemas.enums import DefectType, TerminalOutcome
from src.schemas.models import Defect, DraftAnswer, SqlEvidence
from src.sqlpath.grounding import figures_in, supported_figures, unsupported_figures
from tests.conftest import base_state

AS_OF = date(2025, 12, 31)


def _evidence(**overrides) -> SqlEvidence:
    payload = {
        "template_id": "vendors_by_approval_status",
        "statement": "SELECT ...",
        "parameters": {"approval_status": "Approved", "as_of": str(AS_OF)},
        "row_count": 4,
        "rows": [
            {"vendor_id": 1, "vendor_name": "Northgate Logistics", "risk_category": "Low", "risk_score": 44, "onboarding_date": date(2023, 4, 1)},
            {"vendor_id": 2, "vendor_name": "Vendor_7", "risk_category": "Low", "risk_score": 48, "onboarding_date": date(2024, 2, 9)},
            {"vendor_id": 3, "vendor_name": "Vendor_12", "risk_category": "Medium", "risk_score": 61, "onboarding_date": date(2022, 11, 30)},
            {"vendor_id": 5, "vendor_name": "Vendor_19", "risk_category": "Medium", "risk_score": 55, "onboarding_date": date(2024, 6, 15)},
        ],
        "as_of": AS_OF,
    }
    payload.update(overrides)

    return SqlEvidence(**payload)


# ---------------------------------------------------------------------------------------------
# figures in a records answer are checked against the rows
# ---------------------------------------------------------------------------------------------


def test_figures_are_read_without_dates_citations_and_list_markers():
    text = "As of 2025-12-31 there are 4 vendors:\n1. Vendor_7 (score 48)\n2. Northgate [Vendor Policy §2.1]\n1,200 records"

    assert figures_in(text) == {"4", "48", "1200"}


def test_row_values_counts_per_value_and_the_question_are_supported():
    supported = supported_figures(_evidence(), "vendors onboarded in the last 30 days")

    assert {"4", "44", "61", "2", "2023", "30", "2025", "12", "31"} <= supported


def test_a_figure_the_rows_do_not_hold_is_named():
    answer = "As of 2025-12-31, 4 vendors are approved: 2 Low and 2 Medium risk. The average score is 52 and 36% are new."

    assert unsupported_figures(_evidence(), answer) == ["36", "52"]


def test_an_answer_built_only_from_the_rows_has_no_unsupported_figure():
    answer = "As of 2025-12-31, 4 Low and Medium risk vendors are approved (2 Low, 2 Medium); Vendor_12 scores 61."

    assert unsupported_figures(_evidence(), answer) == []


class _Model:
    def __init__(self, output: ValidatorOutput):
        self.output = output

    def with_structured_output(self, _schema):
        return RunnableLambda(lambda _input, config=None: self.output)


class _Decision:
    tier = "small"

    def as_dict(self):
        return {"node": "compliance_validation", "tier": "small"}


def _validate(monkeypatch, state: dict, output: ValidatorOutput | None) -> dict:
    monkeypatch.setattr(validation, "render_prompt", lambda *_a, **_k: "prompt")
    monkeypatch.setattr(validation, "routed_model", lambda _node, _state: (_Model(output or ValidatorOutput()), _Decision()))

    return validation.compliance_validation_node(state)


def _records_state(answer: str, **overrides) -> dict:
    return base_state(
        "how many vendors are approved and their names",
        standalone_query="how many vendors are approved and their names",
        sql_evidence=_evidence(),
        retrieved_chunks=[],
        draft=DraftAnswer(answer=answer),
        **overrides,
    )


def test_an_invented_figure_fails_deterministically_and_skips_the_validator_model(monkeypatch, no_db, no_tracing):
    def _no_model(_node, _state):
        raise AssertionError("a deterministic failure must not spend a validator call")

    monkeypatch.setattr(validation, "render_prompt", lambda *_a, **_k: "prompt")
    monkeypatch.setattr(validation, "routed_model", _no_model)

    result = validation.compliance_validation_node(_records_state("As of 2025-12-31, 7 vendors are approved."))

    report = result["validation"]
    assert not report.passed
    assert report.defects[0].defect_type is DefectType.SQL_SANITY_FAILURE
    assert "figure 7" in report.defects[0].description
    assert "row count (4)" in report.defects[0].description
    assert result["validation_llm_skipped"]


def test_the_models_grounding_objection_blocks_the_first_pass(monkeypatch, no_db, no_tracing):
    objection = ValidatorOutput(
        defects=[Defect(defect_type=DefectType.SQL_SANITY_FAILURE, description="the count is stated too firmly")]
    )

    result = _validate(monkeypatch, _records_state("As of 2025-12-31, 4 vendors are approved."), objection)

    assert not result["validation"].passed


def test_after_the_rewrite_a_repeated_objection_becomes_a_note_on_a_checked_answer(monkeypatch, no_db, no_tracing):
    objection = ValidatorOutput(
        defects=[
            Defect(defect_type=DefectType.SQL_SANITY_FAILURE, description="the count is stated too firmly"),
            Defect(defect_type=DefectType.UNGROUNDED_CLAIM, description="Vendor_7 is not shown as approved"),
            Defect(defect_type=DefectType.MISSING_CITATION, description="no clause cited"),
        ]
    )
    state = _records_state("As of 2025-12-31, 4 vendors are approved: Vendor_7, Vendor_12, Vendor_19 and Northgate Logistics.", reflection_count=1)

    result = _validate(monkeypatch, state, objection)

    report = result["validation"]
    assert report.passed
    assert [d.defect_type for d in report.defects] == [DefectType.SQL_SANITY_FAILURE, DefectType.UNGROUNDED_CLAIM]
    assert all(d.advisory for d in report.defects)
    assert result["draft"].uncertainty_note.startswith("Validator note: the count is stated too firmly")


def test_the_model_cannot_mark_its_own_defect_advisory(monkeypatch, no_db, no_tracing):
    objection = ValidatorOutput(
        defects=[Defect(defect_type=DefectType.SQL_SANITY_FAILURE, description="x", advisory=True)]
    )

    result = _validate(monkeypatch, _records_state("As of 2025-12-31, 4 vendors are approved."), objection)

    assert not result["validation"].passed


def test_a_rewrite_with_an_unsupported_figure_still_fails(monkeypatch, no_db, no_tracing):
    state = _records_state("As of 2025-12-31, 9 vendors are approved.", reflection_count=1)

    result = _validate(monkeypatch, state, None)

    assert not result["validation"].passed


def test_policy_only_verdicts_never_apply_to_a_records_answer(monkeypatch, no_db, no_tracing):
    output = ValidatorOutput(
        defects=[
            Defect(defect_type=DefectType.POLICY_RULE_BREACH, description="no clause forbids this"),
            Defect(defect_type=DefectType.CLAUSE_CONFLICT, description="none"),
        ]
    )

    result = _validate(monkeypatch, _records_state("As of 2025-12-31, 4 vendors are approved."), output)

    assert result["validation"].passed
    assert result["validation"].defects == []


# ---------------------------------------------------------------------------------------------
# the agentic path runs its workflow inside an event loop
# ---------------------------------------------------------------------------------------------


class _Handler:
    def __await__(self):
        async def _done():
            return "final answer"

        return _done().__await__()


class _Agent:
    def __init__(self):
        self.calls = 0

    def run(self, query, ctx=None):
        self.calls += 1
        # what the real workflow does: schedules itself on the running loop the moment run() is called
        asyncio.get_running_loop()

        return _Handler()


def test_the_agent_workflow_is_started_inside_a_loop_from_a_sync_node(monkeypatch):
    monkeypatch.setattr(agentic_rag, "Context", lambda _agent: None)
    agent = _Agent()

    assert agentic_rag._run_sync(agent, "q") == "final answer"
    assert agent.calls == 1


def test_the_agent_workflow_also_runs_when_a_loop_is_already_running(monkeypatch):
    monkeypatch.setattr(agentic_rag, "Context", lambda _agent: None)

    async def _inside_loop():
        return agentic_rag._run_sync(_Agent(), "q")

    assert asyncio.run(_inside_loop()) == "final answer"


# ---------------------------------------------------------------------------------------------
# the domain gate
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "what is the legal hold process",
        "how are passwords managed for shared accounts",
        "explain the framework for classifying suppliers",
    ],
)
def test_questions_about_the_policies_are_not_refused_on_a_word_list(question):
    ok, _ = is_in_domain(question)

    assert ok


def test_an_unrecognised_subject_is_left_to_the_intent_classifier():
    ok, reason = is_in_domain("tell me how the thing works")

    assert ok
    assert "intent classifier" in reason


def test_clearly_off_topic_requests_are_still_refused():
    ok, _ = is_in_domain("write me a poem about the weather")

    assert not ok


def test_a_records_answer_with_an_advisory_note_is_released(no_db):
    from src.nodes.terminal import output_guardrail_node

    state = base_state(
        "q",
        sql_evidence=_evidence(),
        retrieved_chunks=[],
        draft=DraftAnswer(answer="As of 2025-12-31, 4 vendors are approved.", uncertainty_note="Validator note: x"),
    )

    assert output_guardrail_node(state)["terminal_outcome"] == TerminalOutcome.ANSWERED.value
