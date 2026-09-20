"""Records answers and clause citations (Sep 2026, second batch).

Read out of the production log of 2026-09-11:

* "how many vendors approval status is approved and their names" (Store Manager) escalated after
  72 s: the selector model rejected the vetted query, generated SQL returned 31 rows unscoped, the
  validator saw 10 of them, and two re-plans replayed the same cached draft
* a records answer that passes validation was still rejected at the end for having no clause citation
* generated SQL returned High and Critical vendors to a Store Manager (scope_filtered=0)
* "SET TRANSACTION READ ONLY" never applied on the autocommit pool
* every citation of the privacy policy (§4.1, §5.2 ...) was rejected because the plain-text PDF
  is indexed as one "General §1" section
"""

import time
from contextlib import contextmanager
from datetime import date

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from configs import llms
from configs.settings import settings
from src.auth.rbac import access_scopes_for
from src.graph import routing
from src.guardrails.sql_guard import assert_tables_in_scope, defined_cte_names, referenced_tables
from src.llm_routing import router as router_module
from src.llm_routing.router import RoutingStats
from src.nodes import nl2sql_path as nl2sql_path_module
from src.nodes import rag_path as rag_path_module
from src.nodes.risk_assessment import RiskClassification
from src.nodes.validation import ValidatorOutput
from src.schemas.enums import DefectType, EvidencePath, Intent, RiskLevel, Role, TerminalOutcome
from src.schemas.models import (
    Defect,
    DraftAnswer,
    EvidencePlan,
    IntentResult,
    PlanStep,
    RetrievedChunk,
    SqlEvidence,
    ValidationReport,
)
from src.sqlpath import executor
from src.sqlpath.disclosure import SCOPE_LIMIT, rows_for_prompt, undisclosed, with_disclosures
from src.sqlpath.scoping import ScopeError, scope_statement
from src.sqlpath.selector import match_simple_template
from src.sqlpath.templates import TEMPLATES, templates_visible_to
from tests.conftest import base_state

AS_OF = date(2025, 12, 31)

MANAGER_TABLES = {"vendors", "compliance_reviews"}

OFFICER_TABLES = {"vendors", "compliance_reviews", "audit_logs", "retention_records"}

USER_QUESTION = "how many vendors approval status is approved and their names"

PRIVACY_TITLE = "Retail Data Protection & Customer Privacy Policy"

PRIVACY_TEXT = """Retail Data Protection & Customer Privacy Policy
1. Purpose
This policy establishes standards for the protection of customer personal data.
4. Customer PII Handling
4.1 Data Minimization
Only data strictly required for business operations may be collected.
4.2 Purpose Limitation
PII shall be processed solely for defined, documented business purposes.
5. Consent Management
5.1 Explicit Consent
Consent must be obtained before processing personal data where legally required."""


def _manager_catalogue():
    return templates_visible_to(MANAGER_TABLES)


def _evidence(rows: int = 3, **overrides) -> SqlEvidence:
    payload = {
        "template_id": "vendors_by_approval_status",
        "statement": "WITH vendors AS (...) SELECT vendor_name FROM vendors WHERE approval_status = %(approval_status)s",
        "parameters": {"approval_status": "Approved"},
        "row_count": rows,
        "rows": [
            {"vendor_id": index, "vendor_name": f"Vendor_{index}", "risk_category": "Low", "approval_status": "Approved"}
            for index in range(rows)
        ],
        "as_of": AS_OF,
        "scope_note": "Low and Medium risk vendors only",
        "selection": "matched",
    }
    payload.update(overrides)

    return SqlEvidence(**payload)


# ---------------------------------------------------------------------------------------------
# the plain status question is matched to its vetted query without a model
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "template_id", "parameters"),
    [
        (USER_QUESTION, "vendors_by_approval_status", {"approval_status": "Approved"}),
        ("how many vendors are approved in vendors table", "vendors_by_approval_status", {"approval_status": "Approved"}),
        ("Which vendors are pending approval?", "vendors_by_approval_status", {"approval_status": "Pending"}),
        ("list the rejected suppliers", "vendors_by_approval_status", {"approval_status": "Rejected"}),
        ("show non-compliant vendors and their risk category", "vendors_by_compliance_status", {"compliance_status": "Non-Compliant"}),
        ("which vendors are under review", "vendors_by_compliance_status", {"compliance_status": "Under Review"}),
        ("how many high risk vendors are there", "vendors_by_risk_category", {"risk_category": "High"}),
    ],
)
def test_a_plain_status_question_maps_to_its_vetted_query(question, template_id, parameters):
    matched = match_simple_template(question, _manager_catalogue())

    assert matched is not None
    assert matched[0].template_id == template_id
    assert matched[1] == parameters


@pytest.mark.parametrize(
    "question",
    [
        "which vendors are non compliant and still approved",  # two filters
        "which vendors are not approved",  # negation
        "list unapproved vendors",
        "approved vendors onboarded in 2024",  # a date
        "approved vendors with open findings",  # another table
        "mark all pending vendors as approved",  # an instruction
        "should approved vendors be reviewed every year",  # a policy question
        "how many are approved",  # no vendor noun
        "which vendors are critical",  # a band without the word risk
        "list all vendors",  # no status value: the selector picks the roster query
    ],
)
def test_anything_beyond_a_single_status_filter_still_goes_to_the_selector(question):
    assert match_simple_template(question, _manager_catalogue()) is None


def test_a_named_vendor_is_never_matched_to_a_roster_query():
    assert match_simple_template("is vendor approved", _manager_catalogue(), named_vendor=True) is None


def test_a_template_the_role_cannot_read_is_never_matched():
    assert match_simple_template(USER_QUESTION, templates_visible_to({"compliance_reviews"})) is None


@contextmanager
def _capturing_connection(captured: list, rows: list[dict]):
    class _Result:
        def fetchall(self):
            return rows

    class _Connection:
        def execute(self, statement, parameters=None):
            captured.append((statement, parameters))

            return _Result()

    yield _Connection()


def test_the_user_question_runs_the_vetted_query_with_no_model_call(monkeypatch):
    captured: list = []
    rows = [{"vendor_id": 1, "vendor_name": "Northgate Logistics", "risk_category": "Low", "approval_status": "Approved"}]

    def _no_model(*_args, **_kwargs):
        raise AssertionError("a matched question needs neither the intent model nor the selector")

    monkeypatch.setattr(executor, "get_llm", _no_model)
    monkeypatch.setattr(executor, "select_template", _no_model)
    monkeypatch.setattr(executor, "run_generated_sql", _no_model)
    monkeypatch.setattr(executor, "read_only_connection", lambda: _capturing_connection(captured, rows))

    evidence = executor.run_vetted_sql(
        question=USER_QUESTION, allowed_tables=MANAGER_TABLES, risk_categories={"Low", "Medium"}, as_of=AS_OF
    )

    statement, parameters = captured[0]

    assert evidence.template_id == "vendors_by_approval_status"
    assert evidence.selection == "matched"
    assert parameters["approval_status"] == "Approved"
    assert statement.startswith("WITH vendors AS (SELECT * FROM vendors WHERE risk_category IN ('Low', 'Medium'))")
    assert evidence.statement == statement
    assert evidence.scope_note == "Low and Medium risk vendors only"


def test_a_question_the_matcher_leaves_alone_still_gets_the_intent_model_check(monkeypatch):
    checked: list[str] = []

    def _refuse(_llm, question):
        checked.append(question)

        return False, "Blocked: classified as UPDATE"

    monkeypatch.setattr(executor, "get_llm", lambda _name: None)
    monkeypatch.setattr(executor, "classify_intent_with_llm", _refuse)

    with pytest.raises(executor.SqlPolicyError, match="UPDATE"):
        executor.run_vetted_sql(question="could you get every vendor approved today", allowed_tables=MANAGER_TABLES)

    assert checked


# ---------------------------------------------------------------------------------------------
# the role's scope is written into the statement, so a names-only query or a COUNT is scoped too
# ---------------------------------------------------------------------------------------------


def test_a_store_manager_query_reads_only_low_and_medium_vendors():
    scoped = scope_statement("SELECT vendor_name FROM vendors WHERE approval_status = 'Approved'", MANAGER_TABLES, {"Low", "Medium"})

    assert scoped.statement == (
        "WITH vendors AS (SELECT * FROM vendors WHERE risk_category IN ('Low', 'Medium')) "
        "SELECT vendor_name FROM vendors WHERE approval_status = 'Approved'"
    )
    assert scoped.note == "Low and Medium risk vendors only"


def test_reviews_are_scoped_through_their_vendor_and_defined_before_the_vendor_expression():
    scoped = scope_statement(
        "SELECT reviewer_name, COUNT(*) FROM compliance_reviews GROUP BY reviewer_name", MANAGER_TABLES, {"Low", "Medium"}
    )

    assert scoped.statement.startswith(
        "WITH compliance_reviews AS (SELECT * FROM compliance_reviews WHERE vendor_id IN "
        "(SELECT vendor_id FROM vendors WHERE risk_category IN ('Low', 'Medium')))"
    )
    assert "reviews linked to them" in scoped.note


def test_a_generated_with_query_keeps_its_own_expressions_after_the_scope():
    scoped = scope_statement(
        "WITH approved AS (SELECT vendor_id FROM vendors WHERE approval_status = 'Approved') SELECT COUNT(*) FROM approved",
        MANAGER_TABLES,
        {"Low", "Medium"},
    )

    assert scoped.statement.startswith("WITH vendors AS (")
    assert ", approved AS (SELECT vendor_id FROM vendors" in scoped.statement


def test_a_query_that_redefines_a_scoped_table_or_recurses_is_refused():
    with pytest.raises(ScopeError, match="its own 'vendors'"):
        scope_statement("WITH vendors AS (SELECT 1) SELECT * FROM vendors", MANAGER_TABLES, {"Low"})

    with pytest.raises(ScopeError, match="recursive"):
        scope_statement("WITH RECURSIVE t AS (SELECT 1) SELECT * FROM vendors", MANAGER_TABLES, {"Low"})


def test_a_role_that_sees_every_band_runs_the_statement_unchanged():
    statement = "SELECT vendor_name FROM vendors"

    scoped = scope_statement(statement, OFFICER_TABLES, {"Low", "Medium", "High", "Critical"})

    assert scoped.statement == statement
    assert scoped.note == ""


def test_retention_records_are_scoped_to_the_departments_of_the_user():
    scoped = scope_statement("SELECT * FROM retention_records", OFFICER_TABLES, None, ["finance"])

    assert "retention_records AS (SELECT * FROM retention_records WHERE department IN ('Finance'))" in scoped.statement
    assert "Finance department" in scoped.note


def test_a_table_the_role_was_never_granted_reads_as_empty():
    scoped = scope_statement("SELECT * FROM vendors v, audit_logs a", MANAGER_TABLES, {"Low", "Medium"})

    assert "audit_logs AS (SELECT * FROM audit_logs WHERE FALSE)" in scoped.statement


def test_the_scope_is_applied_to_generated_sql_before_it_runs(monkeypatch):
    from src.sqlpath import nl2sql_engine

    captured: list = []

    monkeypatch.setattr(
        nl2sql_engine, "generate_sql", lambda question, allowed_tables, as_of: "SELECT vendor_name FROM vendors"
    )
    monkeypatch.setattr(nl2sql_engine, "read_only_connection", lambda: _capturing_connection(captured, []))

    evidence = nl2sql_engine.run_generated_sql(
        question="names", allowed_tables=MANAGER_TABLES, risk_categories={"Low", "Medium"}, as_of=AS_OF
    )

    assert captured[0][0].startswith("WITH vendors AS (SELECT * FROM vendors WHERE risk_category IN ('Low', 'Medium'))")
    assert evidence.scope_note == "Low and Medium risk vendors only"
    assert evidence.selection == "generated"


def test_every_vetted_template_still_passes_the_table_check_and_wraps_cleanly():
    for template in TEMPLATES.values():
        statement = " ".join(template.statement.split())

        ok, reason = assert_tables_in_scope(statement, set(template.tables))

        assert ok, (template.template_id, reason)
        assert scope_statement(statement, set(template.tables), {"Low", "Medium"}, ["Finance"]).statement.startswith("WITH ")


# ---------------------------------------------------------------------------------------------
# the table check reads every relation, not just the one right after FROM
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        ("SELECT * FROM vendors v, audit_logs a WHERE a.vendor_id = v.vendor_id", {"vendors", "audit_logs"}),
        ("SELECT * FROM (SELECT * FROM vendors) sub, retention_records r", {"vendors", "retention_records"}),
        ("SELECT * FROM public.vendors", {"public.vendors"}),
        ("SELECT * FROM generate_series(1, 3) g", {"generate_series()"}),
        ("SELECT EXTRACT(YEAR FROM onboarding_date) FROM vendors", {"vendors"}),
        ("SELECT vendor_name FROM vendors WHERE vendor_name ILIKE '%from audit_logs%'", {"vendors"}),
    ],
)
def test_referenced_tables(statement, expected):
    assert referenced_tables(statement) == expected


def test_a_comma_join_to_an_ungranted_table_is_refused():
    ok, reason = assert_tables_in_scope("SELECT v.vendor_name FROM vendors v, audit_logs a", MANAGER_TABLES)

    assert not ok
    assert "audit_logs" in reason


def test_a_query_may_read_its_own_named_expressions():
    statement = "WITH approved AS (SELECT * FROM vendors) SELECT COUNT(*) FROM approved"

    assert defined_cte_names(statement) == {"approved"}
    assert assert_tables_in_scope(statement, MANAGER_TABLES) == (True, "")


def test_the_read_only_connection_runs_inside_a_read_only_transaction(monkeypatch):
    from configs import database

    events: list[str] = []

    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def execute(self, statement):
            events.append(statement)

    class _Transaction:
        def __enter__(self):
            events.append("BEGIN")

        def __exit__(self, *_exc):
            events.append("COMMIT")

            return False

    class _Connection:
        def cursor(self):
            return _Cursor()

        def transaction(self):
            return _Transaction()

        def execute(self, statement, parameters=None):
            events.append(statement)

    class _Pool:
        @contextmanager
        def connection(self):
            yield _Connection()

    monkeypatch.setattr(database, "get_pool", lambda: _Pool())

    with database.read_only_connection() as conn:
        conn.execute("SELECT 1")

    assert events.index("BEGIN") < events.index("SET TRANSACTION READ ONLY") < events.index("SELECT 1")
    assert events[-1] == "COMMIT"


# ---------------------------------------------------------------------------------------------
# the model reads every row with the count in front, and a forgotten caveat is added, not repaired
# ---------------------------------------------------------------------------------------------


def test_the_prompt_rows_carry_the_row_count_and_say_whether_all_are_shown():
    complete = rows_for_prompt(_evidence(rows=31), limit=60)
    partial = rows_for_prompt(_evidence(rows=31), limit=10)

    assert complete.startswith("row_count=31; all 31 rows are listed below; scope: Low and Medium risk vendors only")
    assert "Vendor_30" in complete
    assert partial.startswith("row_count=31; only the first 10 rows are listed below")


def test_a_caveat_the_narration_left_out_is_appended_once():
    evidence = _evidence(generated=True, selection="generated")
    answer = "As of 2025-12-31 there are 3 approved vendors: Vendor_0, Vendor_1 and Vendor_2."

    fixed, added = with_disclosures(evidence, answer)

    assert set(added) == {"generated_query", SCOPE_LIMIT}
    assert fixed.startswith(answer)
    assert undisclosed(evidence, fixed) == []
    assert with_disclosures(evidence, fixed) == (fixed, [])


def test_the_validator_is_given_every_row_and_the_count(monkeypatch, no_db, no_tracing):
    from src.nodes import validation

    seen: dict = {}

    def _render(_name, _fallback, **values):
        seen.update(values)

        return "prompt"

    class _Model:
        def with_structured_output(self, _schema):
            return RunnableLambda(lambda _input, config=None: ValidatorOutput(defects=[], grounded_claim_ratio=1.0))

    class _Decision:
        tier = "small"

        def as_dict(self):
            return {"node": "compliance_validation", "tier": "small"}

    monkeypatch.setattr(validation, "render_prompt", _render)
    monkeypatch.setattr(validation, "routed_model", lambda _node, _state: (_Model(), _Decision()))

    evidence = _evidence(rows=31)
    answer, _ = with_disclosures(evidence, "As of 2025-12-31, 31 vendors are approved.")
    state = base_state(USER_QUESTION, standalone_query=USER_QUESTION, sql_evidence=evidence, draft=DraftAnswer(answer=answer))

    result = validation.compliance_validation_node(state)

    assert result["validation"].passed
    # the validator is told which query produced the rows (its WHERE clause is the evidence for a
    # filter the rows do not show), how many rows there are, and gets all of them
    assert seen["rows"].startswith("query: vetted query vendors_by_approval_status\nSQL: WITH vendors AS")
    assert "parameters: approval_status=Approved" in seen["rows"]
    assert "row_count=31; all 31 rows are listed below" in seen["rows"]
    assert "Vendor_30" in seen["rows"]


# ---------------------------------------------------------------------------------------------
# repair: keep the rows, rewrite the answer once
# ---------------------------------------------------------------------------------------------


def test_a_records_answer_gets_one_repair():
    state = {"evidence_path": "nl2sql", "reflection_count": 0}

    assert routing.repair_attempts_allowed(state) == 1


def test_the_records_repair_keeps_the_rows_and_asks_no_model(monkeypatch, no_db, no_tracing):
    from src.nodes import reflection

    monkeypatch.setattr(reflection, "model_for", lambda _name: (_ for _ in ()).throw(AssertionError("no re-plan model")))

    state = base_state(
        USER_QUESTION,
        standalone_query=USER_QUESTION,
        evidence_path="nl2sql",
        sql_evidence=_evidence(),
        plan=EvidencePlan(path=EvidencePath.NL2SQL, steps=[PlanStep(order=1, source="compliance_db", objective="o", must_prove="p")]),
        validation=ValidationReport(
            passed=False,
            defects=[Defect(defect_type=DefectType.SQL_SANITY_FAILURE, description="the count 31 is not in the rows")],
        ),
    )

    result = reflection.reflection_node(state)

    assert result["repair_strategy"] == routing.SQL_RENARRATE
    assert "sql_evidence" not in result  # the rows stay in the state
    assert result["draft"] is None
    assert "the count 31 is not in the rows" in result["replan_directive"]


FIRST_DRAFT = "As of 2025-12-31, 2 vendors are approved."

REPAIRED_DRAFT = "As of 2025-12-31, 3 vendors in the Low and Medium risk bands are approved: Vendor_0, Vendor_1 and Vendor_2."


class SequencedChatModel:
    """Answers each structured-output schema from a list, one entry per call."""

    def __init__(self, deployment: str, answers: dict, log: list):
        self.deployment = deployment
        self.answers = answers
        self.log = log

    def with_structured_output(self, schema: type[BaseModel]):
        def _answer(messages, config=None):
            queue = self.answers[schema]
            self.log.append((schema.__name__, messages))

            return queue.pop(0) if len(queue) > 1 else queue[0]

        return RunnableLambda(_answer)


@pytest.fixture
def records_graph(memory_graph, monkeypatch):
    log: list = []
    answers = {
        IntentResult: [IntentResult(intent=Intent.VENDOR_STATUS)],
        RiskClassification: [RiskClassification(level=RiskLevel.LOW)],
        DraftAnswer: [DraftAnswer(answer=FIRST_DRAFT), DraftAnswer(answer=REPAIRED_DRAFT)],
        ValidatorOutput: [
            ValidatorOutput(
                defects=[Defect(defect_type=DefectType.SQL_SANITY_FAILURE, description="the answer says 2 but row_count is 3")]
            ),
            ValidatorOutput(defects=[], grounded_claim_ratio=1.0),
        ],
    }
    sql_calls: list = []

    def _run_sql(state):
        sql_calls.append(state["standalone_query"])

        return _evidence(rows=3), ""

    monkeypatch.setattr(llms, "_build_chat_model", lambda spec, temperature: SequencedChatModel(spec.model, answers, log))
    monkeypatch.setattr(nl2sql_path_module, "run_sql_evidence", _run_sql)
    monkeypatch.setattr(router_module, "routing_stats", RoutingStats())
    monkeypatch.setattr(settings, "model_routing_strategy", "heuristic")
    monkeypatch.setattr(settings, "use_model_routing_yaml", False)
    monkeypatch.setattr(settings, "llm_gateway_url", "")

    compiled = memory_graph.get_compiled_graph()
    state = base_state(
        USER_QUESTION,
        role=Role.STORE_MANAGER.value,
        access_scopes=access_scopes_for(Role.STORE_MANAGER),
        thread_id="records",
        request_id="req-records",
        deadline_ts=time.monotonic() + 600,
    )

    final = compiled.invoke(state, config={"configurable": {"thread_id": "records"}, "recursion_limit": 40})

    return final, log, sql_calls


def test_the_store_manager_status_question_is_answered_after_one_rewrite(records_graph):
    final, log, sql_calls = records_graph

    assert final["terminal_outcome"] == TerminalOutcome.ANSWERED.value
    # the query ran once; the repair rewrote the answer over the same rows
    assert len(sql_calls) == 1
    assert final["repair_strategy"] == routing.SQL_RENARRATE
    assert final["reflection_count"] == 1
    # a vetted query answers the question, so the planner model was never asked (the second
    # planner pass is the repair re-using that plan)
    planner_spans = [span for span in final["trace"] if span["node"] == "planner"]
    assert "a vetted query answers it directly" in planner_spans[0]["summary"]
    assert final["path_decision"]["planner"] == "reflection"
    assert not any(name == "EvidencePlan" for name, _ in log)
    # the answer was released without a clause citation, with the scope stated
    assert final["draft"].answer.startswith(REPAIRED_DRAFT)
    assert final["escalation_reason"] is None


def test_the_rewrite_pass_shows_the_writer_what_the_validator_found(records_graph):
    _final, log, _sql_calls = records_graph

    narrations = [messages for name, messages in log if name == "DraftAnswer"]

    assert len(narrations) == 2
    assert not any("the answer says 2 but row_count is 3" in str(message.content) for message in narrations[0])
    assert any("the answer says 2 but row_count is 3" in str(message.content) for message in narrations[1])


def test_the_first_draft_got_the_scope_note_appended_before_validation(records_graph):
    _final, log, _sql_calls = records_graph

    validator_prompts = [messages for name, messages in log if name == "ValidatorOutput"]

    assert "Notes on this result" in str(validator_prompts[0][0].content)


def test_a_records_answer_is_released_without_a_clause_citation(no_db):
    from src.nodes.terminal import output_guardrail_node

    state = base_state(
        USER_QUESTION,
        draft=DraftAnswer(answer="As of 2025-12-31, three Low and Medium risk vendors are approved: A, B and C."),
        sql_evidence=_evidence(),
        retrieved_chunks=[],
    )

    result = output_guardrail_node(state)

    assert result["terminal_outcome"] == TerminalOutcome.ANSWERED.value


def test_a_policy_answer_still_needs_a_clause_citation(no_db):
    from src.nodes.terminal import output_guardrail_node

    chunk = RetrievedChunk(
        chunk_id="c",
        doc_type="privacy_policy",
        document_title=PRIVACY_TITLE,
        section="General",
        clause_number="1",
        version="1.0",
        content=PRIVACY_TEXT,
        rerank_score=0.8,
    )
    state = base_state(
        "what is data minimization",
        draft=DraftAnswer(answer="Only data strictly required for business operations may be collected."),
        retrieved_chunks=[chunk],
    )

    result = output_guardrail_node(state)

    assert result["terminal_outcome"] == TerminalOutcome.ESCALATED.value
    assert "no clause citation" in result["escalation_reason"]


# ---------------------------------------------------------------------------------------------
# a clause heading inside an extract may be cited
# ---------------------------------------------------------------------------------------------


def _privacy_chunk(**overrides) -> RetrievedChunk:
    payload = {
        "chunk_id": "privacy-whole",
        "doc_type": "privacy_policy",
        "document_title": PRIVACY_TITLE,
        "section": "General",
        "clause_number": "1",
        "version": "1.0",
        "content": PRIVACY_TEXT,
        "rerank_score": 0.84,
    }
    payload.update(overrides)

    return RetrievedChunk(**payload)


def test_every_heading_of_a_whole_document_extract_is_citable():
    from src.retrieval.citations import citable_clauses, sub_clause_headings

    numbers = [number for number, _heading, _offset in sub_clause_headings(_privacy_chunk())]
    table = citable_clauses([_privacy_chunk()])

    assert numbers == ["4", "4.1", "4.2", "5", "5.1"]
    assert table["retail data protection customer privacy policy § 4.1"].heading == "Data Minimization"


def test_a_numbered_section_only_lends_its_own_sub_clauses():
    from src.retrieval.citations import sub_clause_headings

    chunk = _privacy_chunk(
        section="Customer PII Handling",
        clause_number="4",
        content="4.1 Data Minimization\nOnly required data.\n1. Customer transaction records\n7. Retention Rules\n",
    )

    assert [number for number, _heading, _offset in sub_clause_headings(chunk)] == ["4.1"]


def test_the_context_header_lists_the_clauses_inside_an_extract():
    from src.retrieval.adapter import format_context

    context = format_context([_privacy_chunk()])

    assert "clauses inside: §4, §4.1, §4.2, §5, §5.1" in context


def test_a_sub_clause_citation_is_grounded_for_the_validator_and_the_output_guard(no_db):
    from src.nodes.terminal import output_guardrail_node
    from src.nodes.validation import _deterministic_defects, validator_extracts

    answer = (
        f"Only data strictly required for business operations may be collected [{PRIVACY_TITLE} §4.1]. "
        f"PII is processed solely for documented purposes [{PRIVACY_TITLE} §4.2]."
    )
    other = _privacy_chunk(chunk_id="other", document_title="Information Security Policy", section="Scope", clause_number="2", content="Scope text.")
    state = base_state("what does the privacy policy say about data minimization", draft=DraftAnswer(answer=answer), retrieved_chunks=[other, _privacy_chunk()])

    assert _deterministic_defects(state) == []
    assert validator_extracts(state)[0].chunk_id == "privacy-whole"
    assert output_guardrail_node(state)["terminal_outcome"] == TerminalOutcome.ANSWERED.value


def test_a_clause_that_is_in_no_extract_is_still_ungrounded():
    from src.nodes.validation import _deterministic_defects

    state = base_state(
        "q",
        draft=DraftAnswer(answer=f"Breaches are reported within 24 hours [{PRIVACY_TITLE} §7.1]."),
        retrieved_chunks=[_privacy_chunk()],
    )

    defects = _deterministic_defects(state)

    assert [defect.defect_type for defect in defects] == [DefectType.UNGROUNDED_CLAIM]


# ---------------------------------------------------------------------------------------------
# Sep 2026: the anti-bribery trace escalated with missing_citation on both attempts even though
# rag_path logged cited=1 both times - the model filled cited_clauses but never wrote the
# [Document §clause] marker into the answer text, so extract_citations(answer) found nothing for
# the deterministic check, the output guardrail, or the compliance validator to see.
# ---------------------------------------------------------------------------------------------


def test_a_cited_clause_missing_its_inline_marker_is_stitched_in():
    from src.nodes.validation import _deterministic_defects
    from src.retrieval.citations import with_inline_citations

    answer = "Only data strictly required for business operations may be collected."

    fixed, added = with_inline_citations(answer, [f"{PRIVACY_TITLE} §4.1"], [_privacy_chunk()])

    assert added == [f"{PRIVACY_TITLE} §4.1"]
    assert fixed == f"{answer[:-1]} [{PRIVACY_TITLE} §4.1]."

    state = base_state("q", draft=DraftAnswer(answer=fixed), retrieved_chunks=[_privacy_chunk()])
    assert _deterministic_defects(state) == []


def test_an_already_inline_citation_is_left_alone():
    from src.retrieval.citations import with_inline_citations

    answer = f"Only data strictly required for business operations may be collected [{PRIVACY_TITLE} §4.1]."

    fixed, added = with_inline_citations(answer, [f"{PRIVACY_TITLE} §4.1"], [_privacy_chunk()])

    assert added == []
    assert fixed == answer


def test_a_cited_clause_the_extracts_never_retrieved_is_not_fabricated():
    from src.retrieval.citations import with_inline_citations

    answer = "Some other claim."

    fixed, added = with_inline_citations(answer, ["Nonexistent Policy §9"], [_privacy_chunk()])

    assert added == []
    assert fixed == answer


def test_the_api_lists_a_cited_sub_clause_with_its_own_text():
    from src.api.routes import _citations_from

    final_state = {
        "retrieved_chunks": [_privacy_chunk()],
        "draft": DraftAnswer(answer="x", cited_clauses=[f"{PRIVACY_TITLE} §4.1"]),
    }

    citations = _citations_from(final_state)

    assert len(citations) == 1
    assert citations[0].clause_number == "4.1"
    assert citations[0].section == "Data Minimization"
    assert citations[0].excerpt.startswith("4.1 Data Minimization")


def test_a_privacy_answer_from_the_old_whole_document_index_is_released(memory_graph, monkeypatch):
    log: list = []
    answer = f"Only data strictly required for business operations may be collected [{PRIVACY_TITLE} §4.1]."
    answers = {
        IntentResult: [IntentResult(intent=Intent.POLICY_LOOKUP)],
        RiskClassification: [RiskClassification(level=RiskLevel.LOW)],
        DraftAnswer: [DraftAnswer(answer=answer, cited_clauses=[f"{PRIVACY_TITLE} §4.1"])],
        ValidatorOutput: [ValidatorOutput(defects=[], grounded_claim_ratio=1.0)],
    }

    monkeypatch.setattr(llms, "_build_chat_model", lambda spec, temperature: SequencedChatModel(spec.model, answers, log))
    monkeypatch.setattr(rag_path_module, "retrieve_policy_evidence", lambda **kwargs: ([_privacy_chunk()], True, [], []))
    monkeypatch.setattr(router_module, "routing_stats", RoutingStats())
    monkeypatch.setattr(settings, "model_routing_strategy", "heuristic")
    monkeypatch.setattr(settings, "use_model_routing_yaml", False)
    monkeypatch.setattr(settings, "llm_gateway_url", "")

    compiled = memory_graph.get_compiled_graph()
    state = base_state(
        "What does the privacy policy say about data minimization?",
        role=Role.STORE_ASSOCIATE.value,
        access_scopes=access_scopes_for(Role.STORE_ASSOCIATE),
        thread_id="privacy",
        request_id="req-privacy",
        deadline_ts=time.monotonic() + 600,
    )

    final = compiled.invoke(state, config={"configurable": {"thread_id": "privacy"}, "recursion_limit": 40})

    assert final["terminal_outcome"] == TerminalOutcome.ANSWERED.value
    assert final["reflection_count"] == 0
    assert final["draft"].cited_clauses == [f"{PRIVACY_TITLE} §4.1"]


# ---------------------------------------------------------------------------------------------
# the workflow view names the new steps
# ---------------------------------------------------------------------------------------------


def test_the_workflow_summary_names_the_matched_query_the_scope_and_the_rewrite():
    from src.observability.agent_steps import describe_step

    nl2sql = describe_step(
        "nl2sql_path",
        {"repair_strategy": routing.SQL_RENARRATE, "reflection_count": 1},
        {"sql_evidence": _evidence(rows=22)},
    )
    reflection = describe_step("reflection", {"repair_strategy": routing.SQL_RENARRATE, "reflection_count": 1}, {})
    planner = describe_step("planner", {"path_decision": {"path": "nl2sql", "planner": "vetted_query"}}, {})

    assert "vendors_by_approval_status · 22 rows · matched without a model call" in nl2sql["summary"]
    assert "scope: Low and Medium risk vendors only" in nl2sql["summary"]
    assert "same rows, answer rewritten" in nl2sql["summary"]
    assert "keep the rows" in reflection["summary"]
    assert reflection["tier"] is None
    assert "vetted query answers it directly" in planner["summary"]


def test_no_hnsw_index_is_attempted_for_embeddings_wider_than_pgvector_allows():
    from src.index.vector_index import hnsw_settings

    assert hnsw_settings(3072) is None
    assert hnsw_settings(1536)["hnsw_m"] == 16


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT query_to_xml('select * from audit_logs', true, false, '') FROM vendors",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT set_config('search_path', 'public', false)",
    ],
)
def test_functions_that_hide_sql_or_touch_the_server_are_refused(statement):
    from src.guardrails.sql_guard import validate_generated_sql

    ok, reason = validate_generated_sql(statement)

    assert not ok
    assert "function" in reason


def test_a_policy_answer_with_no_extracts_and_no_rows_still_needs_a_citation(no_db):
    from src.nodes.terminal import output_guardrail_node

    state = base_state(
        "what is data minimization",
        draft=DraftAnswer(answer="Only data strictly required for business operations may be collected."),
        retrieved_chunks=[],
    )

    assert output_guardrail_node(state)["terminal_outcome"] == TerminalOutcome.ESCALATED.value


def test_every_extract_of_a_cited_clause_reaches_the_validator():
    from src.nodes.validation import validator_extracts

    first_half = _privacy_chunk(chunk_id="half-1")
    second_half = _privacy_chunk(chunk_id="half-2", content="6. Data Sharing Restrictions\n6.1 Third-Party Sharing\nOnly approved vendors.")
    unrelated = [
        _privacy_chunk(chunk_id=f"other-{index}", document_title=f"Other Policy {index}", section="Scope", clause_number="2", content="x")
        for index in range(5)
    ]
    state = base_state(
        "q",
        draft=DraftAnswer(answer=f"Data may only be shared with approved vendors [{PRIVACY_TITLE} §1]."),
        retrieved_chunks=[*unrelated, first_half, second_half],
    )

    chosen = [chunk.chunk_id for chunk in validator_extracts(state)]

    assert chosen[:2] == ["half-1", "half-2"]
