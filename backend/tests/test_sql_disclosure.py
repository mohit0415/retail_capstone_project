from datetime import date

from src.nodes.confidence import confidence_node
from src.schemas.enums import EvidencePath
from src.schemas.models import (
    EvidencePlan,
    PlanStep,
    RetrievedChunk,
    SqlEvidence,
    ValidationReport,
)
from src.sqlpath.disclosure import (
    EMPTY_RESULT,
    ROW_CAP,
    SCOPE_FILTER,
    caveats,
    data_defects,
    disclosures,
    undisclosed,
)

AS_OF = date(2025, 12, 31)


def _evidence(**overrides) -> SqlEvidence:
    payload = {
        "template_id": "vendors_review_overdue",
        "statement": "SELECT 1",
        "parameters": {"as_of": str(AS_OF)},
        "row_count": 3,
        "rows": [{"vendor_id": 1}, {"vendor_id": 2}, {"vendor_id": 3}],
        "as_of": AS_OF,
    }
    payload.update(overrides)

    return SqlEvidence(**payload)


def _plan(source: str) -> EvidencePlan:
    return EvidencePlan(
        path=EvidencePath.NL2SQL,
        steps=[PlanStep(order=1, source=source, objective="o", must_prove="p")],
    )


def test_a_clean_result_needs_no_disclosure():
    assert disclosures(_evidence()) == []


def test_an_empty_result_must_be_disclosed():
    keys = [item.key for item in disclosures(_evidence(row_count=0, rows=[]))]

    assert keys == [EMPTY_RESULT]


def test_a_capped_result_must_be_disclosed():
    keys = [item.key for item in disclosures(_evidence(truncated=True))]

    assert keys == [ROW_CAP]


def test_a_scope_filtered_result_must_be_disclosed():
    keys = [item.key for item in disclosures(_evidence(rows_filtered_by_scope=4))]

    assert keys == [SCOPE_FILTER]


def test_an_answer_that_states_the_empty_result_has_nothing_undisclosed():
    evidence = _evidence(row_count=0, rows=[])
    answer = (
        "As of 2025-12-31 the query returned no rows. An empty result is not evidence "
        "that every review is up to date."
    )

    assert undisclosed(evidence, answer) == []


def test_an_answer_that_hides_the_empty_result_is_caught():
    evidence = _evidence(row_count=0, rows=[])
    answer = "As of 2025-12-31 all vendor reviews are up to date."

    assert [item.key for item in undisclosed(evidence, answer)] == [EMPTY_RESULT]


def test_an_answer_that_hides_the_row_cap_is_caught():
    evidence = _evidence(truncated=True)
    answer = "There are exactly 200 vendors past their review date as of 2025-12-31."

    assert [item.key for item in undisclosed(evidence, answer)] == [ROW_CAP]


def test_an_answer_that_calls_a_capped_count_a_floor_passes():
    evidence = _evidence(truncated=True)
    answer = "At least 200 vendors are past their review date as of 2025-12-31."

    assert undisclosed(evidence, answer) == []


def test_an_answer_that_names_the_scope_filter_passes():
    evidence = _evidence(rows_filtered_by_scope=4)
    answer = "Three vendors match. The result is partial because rows outside your scope were filtered."

    assert undisclosed(evidence, answer) == []


def test_two_missing_disclosures_are_both_reported():
    evidence = _evidence(row_count=0, rows=[], rows_filtered_by_scope=2)
    answer = "Everything looks fine."

    assert [item.key for item in undisclosed(evidence, answer)] == [EMPTY_RESULT, SCOPE_FILTER]


def test_an_observed_date_after_the_as_of_date_is_a_data_defect_not_a_disclosure():
    evidence = _evidence(rows=[{"vendor_id": 1, "last_audit_date": date(2026, 5, 1)}], row_count=1)

    assert disclosures(evidence) == []
    assert data_defects(evidence)


def test_a_due_date_in_the_future_is_not_a_defect():
    evidence = _evidence(
        row_count=2,
        rows=[
            {"vendor_id": 1, "next_review_due": date(2026, 5, 1)},
            {"audit_id": 9, "target_resolution_date": date(2026, 1, 18)},
        ],
    )

    assert data_defects(evidence) == []


def test_caveats_carry_both_disclosures_and_data_defects():
    evidence = _evidence(
        row_count=1,
        rows=[{"vendor_id": 1, "last_audit_date": date(2026, 2, 2)}],
        truncated=True,
    )

    listed = caveats(evidence)

    assert any("row cap" in item for item in listed)
    assert any("after the pinned as_of date" in item for item in listed)


def test_an_honest_empty_sql_answer_scores_above_the_release_threshold():
    state = {
        "retrieved_chunks": [],
        "sql_evidence": _evidence(row_count=0, rows=[]),
        "validation": ValidationReport(passed=True),
        "plan": _plan("compliance_db"),
        "draft": _draft("The query returned no rows as of 2025-12-31."),
    }

    assert confidence_node(state)["confidence"].final_score >= 0.75


def test_a_populated_sql_answer_scores_well_above_the_threshold():
    state = {
        "retrieved_chunks": [],
        "sql_evidence": _evidence(),
        "validation": ValidationReport(passed=True),
        "plan": _plan("compliance_db"),
        "draft": _draft("Three vendors are overdue as of 2025-12-31."),
    }

    assert confidence_node(state)["confidence"].final_score >= 0.85


def test_a_scope_filtered_sql_answer_is_scored_lower_but_still_releasable():
    partial = {
        "retrieved_chunks": [],
        "sql_evidence": _evidence(rows_filtered_by_scope=5),
        "validation": ValidationReport(passed=True),
        "plan": _plan("compliance_db"),
        "draft": _draft("At least three vendors are overdue as of 2025-12-31."),
    }

    whole = {
        **partial,
        "sql_evidence": _evidence(),
    }

    partial_score = confidence_node(partial)["confidence"].final_score
    whole_score = confidence_node(whole)["confidence"].final_score

    assert partial_score < whole_score
    assert partial_score >= 0.75


def test_a_policy_answer_that_was_supposed_to_read_records_is_penalised():
    state = {
        "retrieved_chunks": [_chunk()],
        "sql_evidence": None,
        "validation": ValidationReport(passed=True),
        "plan": _plan("compliance_db"),
        "draft": _draft("The policy requires an annual review [Vendor Policy §4.2]."),
    }

    assert confidence_node(state)["confidence"].source_agreement == 0.7


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        doc_type="vendor_policy",
        document_title="Vendor Policy",
        section="Review",
        clause_number="4.2",
        version="1.0",
        content="Vendors shall be reviewed annually.",
        dense_score=0.6,
    )


def _draft(answer: str):
    from src.schemas.models import DraftAnswer

    return DraftAnswer(answer=answer)
