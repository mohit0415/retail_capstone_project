from datetime import date

import pytest

from src.nodes.confidence import confidence_node
from src.schemas.enums import EvidencePath
from src.schemas.models import DraftAnswer, EvidencePlan, PlanStep, SqlEvidence, ValidationReport
from src.sqlpath import executor, nl2sql_engine
from src.sqlpath.disclosure import GENERATED_QUERY, disclosures, undisclosed
from src.sqlpath.nl2sql_engine import GeneratedSqlError, _gate, _strip
from src.sqlpath.selector import TemplateSelectionError

AS_OF = date(2025, 12, 31)

OFFICER_TABLES = {"vendors", "audit_logs", "retention_records", "compliance_reviews"}

ROSTER_SQL = (
    "SELECT vendor_id, vendor_name, approval_status FROM vendors "
    "ORDER BY approval_status, vendor_name LIMIT 200"
)


def _evidence(**overrides) -> SqlEvidence:
    payload = {
        "template_id": "generated",
        "statement": ROSTER_SQL,
        "parameters": {"as_of": str(AS_OF)},
        "row_count": 2,
        "rows": [
            {"vendor_id": 1, "vendor_name": "Northgate Logistics", "approval_status": "Approved"},
            {"vendor_id": 2, "vendor_name": "Northgate Payments", "approval_status": "Pending"},
        ],
        "as_of": AS_OF,
        "generated": True,
    }
    payload.update(overrides)

    return SqlEvidence(**payload)


# ---------- what comes back from the model ----------


def test_a_fenced_statement_is_unwrapped():
    assert _strip("```sql\nSELECT 1 FROM vendors;\n```") == "SELECT 1 FROM vendors"


def test_a_labelled_statement_is_unwrapped():
    assert _strip("SQL: SELECT 1 FROM vendors") == "SELECT 1 FROM vendors"


def test_a_bare_statement_survives_untouched():
    assert _strip("SELECT 1 FROM vendors") == "SELECT 1 FROM vendors"


# ---------- the gates ----------


def test_a_clean_select_passes_every_gate():
    _gate(ROSTER_SQL, OFFICER_TABLES)


@pytest.mark.parametrize(
    ("statement", "because"),
    [
        ("UPDATE vendors SET approval_status = 'Approved'", "write verb"),
        ("DELETE FROM vendors WHERE vendor_id = 1", "write verb"),
        ("SELECT 1 FROM vendors; DROP TABLE vendors", "stacked statement"),
        ("SELECT 1 FROM vendors -- comment", "sql comment"),
        ("SELECT * FROM vendors WHERE next_review_due < CURRENT_DATE", "clock read"),
        ("SELECT * FROM vendors WHERE next_review_due < NOW()", "clock read"),
        ("WITH x AS (SELECT 1) SELECT * FROM x UNION SELECT * FROM system_audit_log", "union"),
    ],
)
def test_an_unsafe_statement_is_rejected(statement, because):
    with pytest.raises(GeneratedSqlError):
        _gate(statement, OFFICER_TABLES)


def test_a_statement_touching_an_ungranted_table_is_rejected():
    statement = "SELECT * FROM audit_logs LIMIT 10"

    with pytest.raises(GeneratedSqlError):
        _gate(statement, {"vendors"})


def test_a_statement_reading_the_system_audit_log_is_rejected():
    with pytest.raises(GeneratedSqlError):
        _gate("SELECT * FROM system_audit_log LIMIT 10", OFFICER_TABLES)


def test_an_unbound_placeholder_is_rejected_rather_than_run():
    statement = "SELECT * FROM vendors WHERE next_review_due < %(as_of)s LIMIT 200"

    with pytest.raises(GeneratedSqlError, match="placeholder"):
        _gate(statement, OFFICER_TABLES)


# ---------- the two-tier handover ----------


def _install(monkeypatch, *, refuses: bool, generated=None, raises=None):
    monkeypatch.setattr(executor, "validate_question_intent", lambda _q: (True, ""))
    monkeypatch.setattr(executor, "classify_intent_with_llm", lambda _llm, _q: (True, ""))
    monkeypatch.setattr(executor, "get_llm", lambda _name: None)

    if refuses:
        def _select(**_kwargs):
            raise TemplateSelectionError("The catalogue does not provide a query to list all vendors.")

        monkeypatch.setattr(executor, "select_template", _select)

    def _run_generated(**_kwargs):
        if raises is not None:
            raise raises

        return generated

    monkeypatch.setattr(executor, "run_generated_sql", _run_generated)


def test_a_catalogue_miss_falls_through_to_the_generated_engine(monkeypatch):
    _install(monkeypatch, refuses=True, generated=_evidence())

    evidence = executor.run_vetted_sql(question="list all vendors by approval status",
                                       allowed_tables=OFFICER_TABLES, as_of=AS_OF)

    assert evidence.generated
    assert evidence.template_id == "generated"
    assert evidence.row_count == 2


def test_the_fallback_can_be_switched_off(monkeypatch):
    _install(monkeypatch, refuses=True, generated=_evidence())
    monkeypatch.setattr(executor.settings, "enable_generated_sql_fallback", False)

    with pytest.raises(executor.SqlPolicyError, match="catalogue does not provide"):
        executor.run_vetted_sql(question="list all vendors", allowed_tables=OFFICER_TABLES, as_of=AS_OF)


def test_a_generated_failure_reports_both_reasons(monkeypatch):
    _install(monkeypatch, refuses=True, raises=GeneratedSqlError("the generator judged this unanswerable"))

    with pytest.raises(executor.SqlPolicyError) as caught:
        executor.run_vetted_sql(question="list all vendors", allowed_tables=OFFICER_TABLES, as_of=AS_OF)

    assert "catalogue does not provide" in str(caught.value)
    assert "generator judged this unanswerable" in str(caught.value)


def test_a_write_question_never_reaches_the_generator(monkeypatch):
    monkeypatch.setattr(executor, "get_llm", lambda _name: None)
    monkeypatch.setattr(executor, "classify_intent_with_llm", lambda _llm, _q: (True, ""))

    def _explode(**_kwargs):
        raise AssertionError("a blocked question must not reach the generated engine")

    monkeypatch.setattr(executor, "run_generated_sql", _explode)

    with pytest.raises(executor.SqlPolicyError, match="Blocked"):
        executor.run_vetted_sql(question="delete all vendor records", allowed_tables=OFFICER_TABLES)


def test_generated_rows_are_still_scoped_by_department(monkeypatch):
    rows = [
        {"retention_id": 1, "department": "Finance"},
        {"retention_id": 2, "department": "IT"},
    ]
    _install(monkeypatch, refuses=True, generated=_evidence(rows=rows, row_count=2))

    evidence = executor.run_vetted_sql(
        question="list every retention record",
        allowed_tables=OFFICER_TABLES,
        departments=["Finance"],
        as_of=AS_OF,
    )

    assert evidence.row_count == 1
    assert evidence.rows_filtered_by_scope == 1


def test_a_role_with_no_table_never_reaches_the_generator(monkeypatch):
    monkeypatch.setattr(executor, "validate_question_intent", lambda _q: (True, ""))
    monkeypatch.setattr(executor, "classify_intent_with_llm", lambda _llm, _q: (True, ""))
    monkeypatch.setattr(executor, "get_llm", lambda _name: None)

    def _explode(**_kwargs):
        raise AssertionError("a role with no grant must not reach the generated engine")

    monkeypatch.setattr(executor, "run_generated_sql", _explode)

    with pytest.raises(executor.SqlPolicyError, match="not scoped to any compliance table"):
        executor.run_vetted_sql(question="list all vendors", allowed_tables=set())


def test_the_engine_refuses_a_role_with_no_table():
    with pytest.raises(GeneratedSqlError, match="not scoped"):
        nl2sql_engine.run_generated_sql(question="anything", allowed_tables=set())


# ---------- what the answer has to say about it ----------


def test_a_generated_result_must_be_disclosed():
    assert [item.key for item in disclosures(_evidence())] == [GENERATED_QUERY]


def test_an_answer_that_names_the_generated_query_passes():
    answer = (
        "As of 2025-12-31 there are two vendors. No reviewed query covered this question, so the "
        "SQL was generated for it."
    )

    assert undisclosed(_evidence(), answer) == []


def test_an_answer_that_hides_the_generated_query_is_caught():
    answer = "As of 2025-12-31 there are two vendors: Northgate Logistics and Northgate Payments."

    assert [item.key for item in undisclosed(_evidence(), answer)] == [GENERATED_QUERY]


def test_a_generated_answer_still_releases_but_scores_below_a_vetted_one():
    def _state(evidence):
        return {
            "retrieved_chunks": [],
            "sql_evidence": evidence,
            "validation": ValidationReport(passed=True),
            "plan": EvidencePlan(
                path=EvidencePath.NL2SQL,
                steps=[PlanStep(order=1, source="compliance_db", objective="o", must_prove="p")],
            ),
            "draft": DraftAnswer(answer="Two vendors, from a generated query, as of 2025-12-31."),
        }

    generated = confidence_node(_state(_evidence()))["confidence"].final_score
    vetted = confidence_node(_state(_evidence(generated=False, template_id="vendors_all")))
    vetted_score = vetted["confidence"].final_score

    assert generated < vetted_score
    assert generated >= 0.75
