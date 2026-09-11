"""A hybrid question asks the records query only its records half, cites its policy sentences, and is
repaired without an LLM re-plan.

Read out of the log: "How often must a Low risk vendor be monitored under the vendor policy, and which
of our vendors are Low risk?" went to human review after three attempts and 93 s. Given the whole
question, the template selector said the question was about policy and no vetted query fitted, so
generated SQL ran (21 rows, then 28); the monitoring sentence carried no citation, so the validator
failed the draft; and each LLM re-plan sent the same question through the same stages again.
"""

from datetime import date

import pytest

from src.graph import routing
from src.schemas.enums import DefectType
from src.schemas.models import Defect, DraftAnswer, RetrievedChunk, SqlEvidence, ValidationReport
from tests.conftest import base_state

SCOPES = ["doc:vendor_policy", "table:vendors", "table:compliance_reviews", "risk:High"]

QUESTION = "How often must a Low risk vendor be monitored under the vendor policy, and which of our vendors are Low risk?"

POLICY_TITLE = "Supplier Vendor Compliance Policy"

MONITORING = RetrievedChunk(
    chunk_id="vendor-4",
    doc_type="vendor_policy",
    document_title=POLICY_TITLE,
    section="Ongoing Monitoring & Escalation",
    clause_number="4",
    version="1.0",
    content=(
        "Low risk vendors require an annual confirmation of compliance. Medium risk vendors are reviewed "
        "bi-annually. High risk vendors receive a quarterly compliance audit."
    ),
    rerank_score=0.38,
)

ROWS = SqlEvidence(
    template_id="vendors_by_risk_category",
    statement="SELECT vendor_name FROM vendors WHERE risk_category = %(risk_category)s",
    parameters={"risk_category": "Low"},
    row_count=2,
    rows=[{"vendor_name": "Vendor_1"}, {"vendor_name": "Vendor_2"}],
    as_of=date(2025, 12, 31),
    selection="matched",
)


class _Decision:
    tier = "strong"

    def as_dict(self):
        return {"node": "stub", "tier": "strong"}


def _writer(answer: str, prompts: list | None = None):
    class _Model:
        def with_structured_output(self, _schema):
            return self

        def invoke(self, messages, config=None):
            if prompts is not None:
                prompts.append(messages)

            return DraftAnswer(answer=answer)

    return lambda node, state: (_Model(), _Decision())


# ---------------------------------------------------------------------------------------------
# the records query gets only the records half
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (QUESTION, "which of our vendors are Low risk?"),
        (
            "What is the retention period for customer transaction records, and which Finance retention records do we hold?",
            "which Finance retention records do we hold?",
        ),
        (
            "What controls does the vendor policy require for Medium risk vendors, and which vendors are in the Medium band?",
            "which vendors are in the Medium band?",
        ),
        ("Which vendors are currently approved?", "Which vendors are currently approved?"),
        ("Is vendor 7 approved and what does the policy require?", "Is vendor 7 approved and what does the policy require?"),
    ],
)
def test_the_records_half_of_a_two_part_question(question, expected):
    from src.nodes.nl2sql_path import records_part

    assert records_part(question) == expected


def test_the_records_half_matches_the_vetted_template_without_a_model_call():
    from src.nodes.nl2sql_path import records_part
    from src.sqlpath.selector import match_simple_template
    from src.sqlpath.templates import TEMPLATES

    templates = list(TEMPLATES.values())

    matched = match_simple_template(records_part(QUESTION), templates)

    assert matched is not None
    assert matched[0].template_id == "vendors_by_risk_category"
    assert "Low" in str(matched[1])
    # the whole question never matched, which is what sent it to generated SQL
    assert match_simple_template(QUESTION, templates) is None


def test_the_hybrid_stage_asks_the_records_query_only_the_records_half(monkeypatch, no_db, no_tracing):
    from src.nodes import hybrid_path, nl2sql_path

    asked: list[str] = []

    def _run(state):
        asked.append(state["standalone_query"])

        return ROWS, ""

    monkeypatch.setattr(nl2sql_path, "run_sql_evidence", _run)
    monkeypatch.setattr(
        hybrid_path,
        "routed_model",
        _writer(f"Low risk vendors require an annual confirmation of compliance [{POLICY_TITLE} §4]."),
    )

    state = base_state(QUESTION, access_scopes=SCOPES, routed_path="hybrid", retrieved_chunks=[MONITORING])
    result = nl2sql_path.nl2sql_path_node(state)

    assert asked == ["which of our vendors are Low risk?"]
    assert result["sql_evidence"] is ROWS
    # the writer still answers the whole question
    assert state["standalone_query"] == QUESTION


# ---------------------------------------------------------------------------------------------
# the policy sentences of a hybrid answer carry their clause; the rows sentences do not
# ---------------------------------------------------------------------------------------------


def test_the_policy_sentence_gets_its_clause_and_the_rows_sentence_does_not(monkeypatch, no_db, no_tracing):
    from src.nodes import hybrid_path

    monkeypatch.setattr(
        hybrid_path,
        "routed_model",
        _writer(
            "Low risk vendors must be monitored with an annual confirmation of compliance. "
            "As of 2025-12-31, 2 vendors are rated Low risk: Vendor_1 and Vendor_2."
        ),
    )

    state = base_state(QUESTION, access_scopes=SCOPES, routed_path="hybrid")
    draft, _decision, _caveats = hybrid_path.draft_hybrid_answer(state, [MONITORING], ROWS, "", 60)

    assert f"annual confirmation of compliance [{POLICY_TITLE} §4]." in draft.answer
    assert "Vendor_2 [" not in draft.answer
    assert f"{POLICY_TITLE} §4" in draft.cited_clauses


def test_a_record_sentence_is_recognised():
    from src.retrieval.citations import is_record_sentence

    assert is_record_sentence("As of 2025-12-31, 21 vendors are rated Low risk.")
    assert is_record_sentence("The Low risk vendors are Vendor_1 and Vendor_2.")
    assert is_record_sentence("The following vendors are classified as Low risk:")
    assert not is_record_sentence("Low risk vendors require an annual confirmation of compliance.")


# ---------------------------------------------------------------------------------------------
# a hybrid repair keeps the rows and makes no model call
# ---------------------------------------------------------------------------------------------


def test_a_hybrid_repair_keeps_the_rows_and_makes_no_model_call(monkeypatch, no_db, no_tracing):
    from src.nodes import reflection

    monkeypatch.setattr(reflection, "model_for", lambda _name: pytest.fail("reflection model called for a hybrid repair"))

    state = base_state(
        QUESTION,
        access_scopes=SCOPES,
        routed_path="hybrid",
        evidence_path="hybrid",
        sql_evidence=ROWS,
        validation=ValidationReport(
            passed=False,
            defects=[Defect(defect_type=DefectType.MISSING_CITATION, description="the monitoring sentence has no citation")],
        ),
    )

    result = reflection.reflection_node(state)

    assert result["repair_strategy"] == routing.HYBRID_REDRAFT
    assert "sql_evidence" not in result
    assert result["retrieved_chunks"] is None
    assert "the monitoring sentence has no citation" in result["replan_directive"]


def test_the_hybrid_repair_pass_rewrites_the_answer_over_the_same_rows(monkeypatch, no_db, no_tracing):
    from src.nodes import hybrid_path, nl2sql_path

    prompts: list = []

    monkeypatch.setattr(nl2sql_path, "run_sql_evidence", lambda state: pytest.fail("the records were queried again"))
    monkeypatch.setattr(
        hybrid_path,
        "routed_model",
        _writer(f"Low risk vendors require an annual confirmation of compliance [{POLICY_TITLE} §4].", prompts),
    )

    state = base_state(
        QUESTION,
        access_scopes=SCOPES,
        routed_path="hybrid",
        retrieved_chunks=[MONITORING],
        sql_evidence=ROWS,
        repair_strategy=routing.HYBRID_REDRAFT,
        reflection_count=1,
        replan_directive="Fix every defect: cite the monitoring sentence",
    )

    result = nl2sql_path.nl2sql_path_node(state)

    assert result["sql_evidence"] is ROWS
    assert any("cite the monitoring sentence" in str(message.content) for message in prompts[0])


# ---------------------------------------------------------------------------------------------
# the validator's missing_citation on a hybrid answer
# ---------------------------------------------------------------------------------------------


def _judge(claim: str):
    from src.nodes import validation

    class _Model:
        def with_structured_output(self, _schema):
            return self

        def invoke(self, messages, config=None):
            return validation.ValidatorOutput(
                defects=[Defect(defect_type=DefectType.MISSING_CITATION, description="no citation", offending_claim=claim)],
                grounded_claim_ratio=0.8,
            )

    return lambda node, state: (_Model(), _Decision())


def _hybrid_state(answer: str) -> dict:
    return base_state(
        QUESTION,
        access_scopes=SCOPES,
        routed_path="hybrid",
        retrieved_chunks=[MONITORING],
        sql_evidence=ROWS,
        draft=DraftAnswer(answer=answer),
    )


@pytest.mark.parametrize(
    "claim",
    ["Low risk vendors require an annual confirmation", "As of 2025-12-31 the Low risk vendors are Vendor_1"],
)
def test_a_missing_citation_on_a_cited_or_rows_sentence_of_a_hybrid_answer_is_a_note(claim, monkeypatch, no_db, no_tracing):
    from src.nodes import validation

    monkeypatch.setattr(validation, "routed_model", _judge(claim))
    monkeypatch.setattr(validation, "data_defects", lambda evidence: [])
    monkeypatch.setattr(validation, "undisclosed", lambda evidence, answer: [])

    answer = (
        f"Low risk vendors require an annual confirmation of compliance [{POLICY_TITLE} §4]. "
        "As of 2025-12-31 the Low risk vendors are Vendor_1 and Vendor_2."
    )

    report = validation.compliance_validation_node(_hybrid_state(answer))["validation"]

    assert report.passed
    assert report.defects[0].advisory


def test_a_missing_citation_on_an_uncited_policy_sentence_of_a_hybrid_answer_still_blocks(monkeypatch, no_db, no_tracing):
    from src.nodes import validation

    monkeypatch.setattr(validation, "routed_model", _judge("Low risk vendors require an annual confirmation"))
    monkeypatch.setattr(validation, "data_defects", lambda evidence: [])
    monkeypatch.setattr(validation, "undisclosed", lambda evidence, answer: [])

    answer = (
        "Low risk vendors require an annual confirmation of compliance. "
        "As of 2025-12-31 the Low risk vendors are Vendor_1 and Vendor_2."
    )

    assert not validation.compliance_validation_node(_hybrid_state(answer))["validation"].passed


# ---------------------------------------------------------------------------------------------
# the confidence of a hybrid answer does not hang on the validator's grounded ratio
# ---------------------------------------------------------------------------------------------

LIVE_ANSWER = (
    f"A Low risk vendor must be monitored with an Annual confirmation [{POLICY_TITLE} §4].\n\n"
    "The following vendors are classified as Low risk:\n"
    "1. Vendor_28 (Compliance Status: Under Review)\n"
    "2. Vendor_6 (Compliance Status: Compliant)\n"
    "3. Vendor_1 (Compliance Status: Non-Compliant)"
)


def test_only_the_policy_sentences_of_a_hybrid_answer_count_for_citation_grounding():
    from src.retrieval.citations import citation_grounding

    assert citation_grounding(LIVE_ANSWER, [MONITORING], skip_records=True) == 1.0
    # counted, the vendor lines drowned the one cited policy sentence
    assert citation_grounding(LIVE_ANSWER, [MONITORING]) < 0.5


@pytest.mark.parametrize("judged", [0.0, 0.5, 0.86])
def test_a_hybrid_answer_grounding_is_the_same_whatever_ratio_the_validator_gives(judged):
    from src.nodes.confidence import _grounding

    state = base_state(
        QUESTION,
        access_scopes=SCOPES,
        routed_path="hybrid",
        retrieved_chunks=[MONITORING],
        sql_evidence=ROWS,
        draft=DraftAnswer(answer=LIVE_ANSWER),
        validation=ValidationReport(passed=True, defects=[], grounded_claim_ratio=judged),
    )

    assert _grounding(state, [MONITORING]) == 1.0
