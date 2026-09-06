import pytest

from src.auth.rbac import (
    access_scopes_for,
    allowed_doc_types,
    allowed_risk_categories,
    allowed_tables,
)
from src.guardrails.sql_guard import (
    assert_tables_in_scope,
    validate_generated_sql,
    validate_question_intent,
)
from src.schemas.enums import Role
from src.schemas.models import SqlEvidence
from src.sqlpath.executor import _scope_rows, sanity_check
from src.sqlpath.templates import (
    TABLE_DESCRIPTIONS,
    TEMPLATES,
    schema_notes_for,
    tables_visible_to,
)


def test_associate_sees_no_tables():
    scopes = access_scopes_for(Role.STORE_ASSOCIATE)

    assert allowed_tables(scopes) == set()
    assert tables_visible_to(allowed_tables(scopes)) == []


def test_manager_cannot_read_audit_or_retention():
    tables = allowed_tables(access_scopes_for(Role.STORE_MANAGER))

    assert "audit_logs" not in tables
    assert "retention_records" not in tables
    assert "vendors" in tables


def test_manager_is_capped_at_medium_risk_vendors():
    categories = allowed_risk_categories(access_scopes_for(Role.STORE_MANAGER))

    assert categories == {"Low", "Medium"}
    assert "Critical" not in categories


def test_compliance_officer_sees_every_risk_band():
    categories = allowed_risk_categories(access_scopes_for(Role.COMPLIANCE_OFFICER))

    assert categories == {"Low", "Medium", "High", "Critical"}


def test_compliance_officer_sees_the_regulatory_documents():
    documents = allowed_doc_types(access_scopes_for(Role.COMPLIANCE_OFFICER))

    assert "gdpr" in documents
    assert "iso_27001" in documents


def test_associate_cannot_see_the_vendor_policy():
    assert "vendor_policy" not in allowed_doc_types(access_scopes_for(Role.STORE_ASSOCIATE))


def test_the_four_domain_tables_are_described():
    assert set(TABLE_DESCRIPTIONS) == {
        "vendors",
        "audit_logs",
        "retention_records",
        "compliance_reviews",
    }


def test_system_audit_log_is_not_reachable_by_nl2sql():
    for role in Role:
        assert "system_audit_log" not in tables_visible_to(allowed_tables(access_scopes_for(role)))


def test_schema_notes_only_describe_tables_in_scope():
    notes = schema_notes_for({"vendors"})

    assert "vendors:" in notes
    assert "retention_records:" not in notes


def test_schema_notes_carry_the_exact_stored_value_casing():
    notes = schema_notes_for({"vendors", "audit_logs"})

    assert "'Non-Compliant'" in notes or "Non-Compliant" in notes
    assert "Under Review" in notes
    assert "In Progress" in notes


def test_schema_notes_pin_the_as_of_date_and_disclaim_the_clock():
    notes = schema_notes_for({"vendors"})

    assert "2025-12-31" in notes
    assert "database clock" in notes


def test_no_vetted_template_reads_the_wall_clock():
    clock_functions = ("current_date", "now(", "current_timestamp", "localtimestamp")

    for template in TEMPLATES.values():
        statement = template.statement.lower()

        for function in clock_functions:
            assert function not in statement, f"{template.template_id} reads the clock via {function}"


def test_schema_notes_separate_the_two_status_columns():
    notes = schema_notes_for({"vendors"})

    assert "independent" in notes


def test_schema_notes_explain_that_retention_period_is_a_duration():
    notes = schema_notes_for({"retention_records"})

    assert "duration" in notes


@pytest.mark.parametrize(
    "question",
    [
        "delete the retention records for the Finance department",
        "please remove that vendor row",
        "update vendors set compliance_status = 'Compliant'",
        "drop table audit_logs",
        "mark Vendor_8 as Compliant and close its open findings",
    ],
)
def test_mutation_questions_are_refused(question):
    ok, reason = validate_question_intent(question)

    assert not ok
    assert reason


@pytest.mark.parametrize(
    "question",
    [
        "which vendors are in the Critical risk category",
        "how many findings are still open past their target date",
        "show the compliance reviews for Vendor_42",
    ],
)
def test_read_only_questions_pass(question):
    ok, _ = validate_question_intent(question)

    assert ok


def test_generated_write_sql_is_rejected():
    ok, reason = validate_generated_sql("DELETE FROM vendors WHERE vendor_id = 1")

    assert not ok
    assert reason


def test_generated_sql_reading_the_clock_is_rejected():
    ok, reason = validate_generated_sql(
        "SELECT vendor_id FROM vendors WHERE next_review_due < CURRENT_DATE LIMIT 10"
    )

    assert not ok
    assert "wall clock" in reason


def test_stacked_statement_is_rejected():
    ok, _ = validate_generated_sql("SELECT 1 FROM vendors; DROP TABLE vendors")

    assert not ok


def test_union_injection_is_rejected():
    ok, _ = validate_generated_sql("SELECT vendor_id FROM vendors UNION SELECT password FROM users")

    assert not ok


def test_clean_pinned_select_is_accepted():
    ok, _ = validate_generated_sql(
        "SELECT vendor_id, vendor_name FROM vendors WHERE next_review_due < '2025-12-31' LIMIT 50"
    )

    assert ok


def test_cte_select_is_accepted():
    ok, _ = validate_generated_sql(
        "WITH overdue AS (SELECT audit_id FROM audit_logs WHERE target_resolution_date < '2025-12-31') "
        "SELECT COUNT(*) FROM overdue LIMIT 1"
    )

    assert ok


def test_out_of_scope_table_reference_is_caught():
    statement = "SELECT a.audit_id FROM audit_logs a JOIN vendors v ON v.vendor_id = a.vendor_id LIMIT 10"

    ok, reason = assert_tables_in_scope(statement, {"vendors"})

    assert not ok
    assert "audit_logs" in reason


def test_in_scope_join_is_allowed():
    statement = "SELECT cr.review_id FROM compliance_reviews cr JOIN vendors v ON v.vendor_id = cr.vendor_id"

    ok, _ = assert_tables_in_scope(statement, {"vendors", "compliance_reviews"})

    assert ok


def test_risk_scope_removes_vendors_above_the_role_band():
    rows = [
        {"vendor_id": 1, "risk_category": "Low"},
        {"vendor_id": 2, "risk_category": "Critical"},
        {"vendor_id": 3, "risk_category": "Medium"},
    ]

    kept, removed = _scope_rows(rows, departments=[], risk_categories={"Low", "Medium"})

    assert [row["vendor_id"] for row in kept] == [1, 3]
    assert removed == 1


def test_department_scope_removes_other_departments():
    rows = [{"department": "Finance"}, {"department": "HR"}]

    kept, removed = _scope_rows(rows, departments=["Finance"], risk_categories=None)

    assert len(kept) == 1
    assert removed == 1


def test_rows_without_the_scoped_column_survive():
    rows = [{"audit_id": 1, "issue_severity": "High"}]

    kept, removed = _scope_rows(rows, departments=["Finance"], risk_categories={"Low"})

    assert kept == rows
    assert removed == 0


def test_empty_result_is_flagged_as_not_evidence_of_compliance():
    from datetime import date

    evidence = SqlEvidence(
        template_id="nl2sql_engine",
        statement="SELECT 1",
        parameters={},
        row_count=0,
        rows=[],
        as_of=date(2025, 12, 31),
    )

    assert any("not evidence of compliance" in item for item in sanity_check(evidence))


def test_scope_filtered_rows_are_flagged_as_partial():
    from datetime import date

    evidence = SqlEvidence(
        template_id="nl2sql_engine",
        statement="SELECT 1",
        parameters={},
        row_count=2,
        rows=[{"vendor_id": 1}, {"vendor_id": 2}],
        as_of=date(2025, 12, 31),
        rows_filtered_by_scope=5,
    )

    assert any("partial" in item for item in sanity_check(evidence))


@pytest.mark.parametrize(
    "question",
    [
        "how do we close a finding once remediation is complete",
        "what does it mean to mark a vendor as Non-Compliant",
        "which vendors have a Closed review status",
        "explain how approval status changes over time",
        "show me the remediation status of every open finding",
    ],
)
def test_procedure_questions_are_not_mistaken_for_write_intent(question):
    ok, reason = validate_question_intent(question)

    assert ok, reason


@pytest.mark.parametrize(
    "question",
    [
        "mark Vendor_8 as Compliant",
        "close its open findings",
        "flag the review as complete",
        "reject the approval for Vendor_3",
    ],
)
def test_imperative_status_changes_are_refused(question):
    ok, _ = validate_question_intent(question)

    assert not ok


def test_human_escalation_phrasing_still_reaches_the_handoff():
    ok, _ = validate_question_intent("escalate this to a human reviewer please")

    assert ok
