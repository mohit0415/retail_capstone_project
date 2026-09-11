"""``src/api/escalation_service.py`` against a fake connection, plus the two
small observability helpers that had no tests: the in-process ``SloRecorder``
and the ``LoggingCallbackHandler`` attached to every model call.
"""

import json
import logging
from contextlib import contextmanager
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from src.api import escalation_service
from src.core.slo import SloRecorder
from src.observability.callbacks import LoggingCallbackHandler, _token_usage
from src.schemas.enums import ReviewDecision


class _Cursor:
    def __init__(self, rows, rowcount=None):
        self.rows = rows
        self.rowcount = len(rows) if rowcount is None else rowcount

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeConnection:
    def __init__(self, rows=None, rowcount=0):
        self.rows = rows or []
        self.rowcount = rowcount
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

        return _Cursor(self.rows, self.rowcount)


@pytest.fixture
def db(monkeypatch):
    def _install(rows=None, rowcount=0):
        conn = FakeConnection(rows, rowcount)

        @contextmanager
        def _connection():
            yield conn

        monkeypatch.setattr(escalation_service, "read_only_connection", _connection)
        monkeypatch.setattr(escalation_service, "writable_connection", _connection)

        return conn

    return _install


def _queue_row(**overrides) -> dict:
    row = {
        "request_id": "req-1",
        "thread_id": "thread-1",
        "user_id": "asker",
        "risk_level": "High",
        "reason": "risk level is High",
        "status": "pending",
        "context_package": {
            "original_query": "may I accept this gift?",
            "standalone_query": "may I accept a gift from vendor Acme?",
            "draft_answer": "No [Anti Bribery Policy §3.1].",
            "confidence": 0.55,
            "evidence_path": "high_risk_panel",
            "panel_verdict": {"consensus": "No", "unresolved_conflict": False},
            "reflection_count": 1,
        },
        "reviewer_decision": None,
        "reviewed_answer": None,
        "queued_at": datetime.now(UTC),
        "reviewed_at": None,
    }
    row.update(overrides)

    return row


def test_list_pending_maps_rows_to_queue_items_with_the_limit(db):
    conn = db(rows=[_queue_row(), _queue_row(request_id="req-2", risk_level="Medium")])

    items = escalation_service.list_pending(limit=7)

    assert [item.request_id for item in items] == ["req-1", "req-2"]
    assert items[1].risk_level == "Medium"
    assert conn.executed[0][1] == {"limit": 7}
    assert "status = 'pending'" in conn.executed[0][0]


def test_fetch_package_returns_nothing_for_an_unknown_request(db):
    db(rows=[])

    assert escalation_service.fetch_package("nope") == (None, None)


def test_fetch_package_builds_the_review_from_the_stored_context(db):
    db(rows=[_queue_row()])

    review, row = escalation_service.fetch_package("req-1")

    assert review.request_id == "req-1"
    assert review.original_query == "may I accept this gift?"
    assert review.standalone_query.startswith("may I accept a gift")
    assert review.draft_answer == "No [Anti Bribery Policy §3.1]."
    assert review.confidence == 0.55
    assert review.evidence_path == "high_risk_panel"
    assert review.panel_verdict == {"consensus": "No", "unresolved_conflict": False}
    assert review.reflection_count == 1
    assert review.panel_repair_count == 0
    assert review.conversation_history == []
    assert row["status"] == "pending"


def test_fetch_package_accepts_a_json_string_column_too(db):
    db(rows=[_queue_row(context_package=json.dumps({"draft_answer": "text", "confidence": None}))])

    review, _ = escalation_service.fetch_package("req-1")

    assert review.draft_answer == "text"
    assert review.confidence == 0.0
    assert review.validation_output == {}


def test_record_decision_updates_only_a_pending_row(db):
    conn = db(rowcount=1)

    updated = escalation_service.record_decision(
        request_id="req-1",
        reviewer_id="rev",
        decision=ReviewDecision.EDIT,
        edited_answer="Yes, up to the gift limit [Anti Bribery Policy §3.2].",
        notes="clarified the threshold",
    )

    assert updated is True
    sql, params = conn.executed[0]
    assert "AND status = 'pending'" in sql
    assert params["decision"] == "edit"
    assert params["answer"].startswith("Yes, up to")
    assert params["reviewer_id"] == "rev"
    assert params["notes"] == "clarified the threshold"
    assert params["reviewed_at"].tzinfo is not None


def test_record_decision_reports_a_lost_race(db):
    db(rowcount=0)

    assert escalation_service.record_decision("req-1", "rev", ReviewDecision.ACCEPT, "draft", "") is False


def test_queue_depth_reads_the_count_and_tolerates_no_row(db):
    db(rows=[{"depth": 3}])
    assert escalation_service.queue_depth() == 3

    db(rows=[])
    assert escalation_service.queue_depth() == 0


def test_an_empty_recorder_reports_no_percentiles_and_no_verdict():
    snap = SloRecorder().snapshot(p85_target_ms=100, p95_target_ms=200, minimum_samples=5)

    assert snap["sample_count"] == 0
    assert snap["p85_ms"] is None
    assert snap["p95_ms"] is None
    assert snap["p85_within_slo"] is False
    assert snap["sufficient_sample_size"] is False


def test_a_single_sample_is_its_own_percentile():
    recorder = SloRecorder()
    recorder.observe(42.0)

    snap = recorder.snapshot(100, 200, minimum_samples=1)

    assert snap["p85_ms"] == snap["p95_ms"] == 42.0
    assert snap["p95_within_slo"] is True


def test_percentiles_and_targets_are_compared_once_enough_samples_arrived():
    recorder = SloRecorder()

    for value in range(1, 101):
        recorder.observe(float(value))

    snap = recorder.snapshot(p85_target_ms=90, p95_target_ms=90, minimum_samples=50)

    assert snap["sample_count"] == 100
    assert 84 <= snap["p85_ms"] <= 86
    assert 94 <= snap["p95_ms"] <= 96
    assert snap["p85_within_slo"] is True
    assert snap["p95_within_slo"] is False


def test_the_recorder_is_bounded_and_clamps_negative_latencies():
    recorder = SloRecorder(max_samples=3)

    for value in (-5.0, 10.0, 20.0, 30.0):
        recorder.observe(value)

    snap = recorder.snapshot(100, 100, minimum_samples=1)

    assert snap["sample_count"] == 3
    assert snap["p95_ms"] <= 30.0
    assert snap["p85_ms"] >= 10.0


def _result(llm_output=None, usage_metadata=None) -> LLMResult:
    message = AIMessage(content="ok", usage_metadata=usage_metadata)

    return LLMResult(generations=[[ChatGeneration(message=message)]], llm_output=llm_output)


def test_token_usage_is_read_from_llm_output_first():
    usage = _token_usage(
        _result({"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
    )

    assert usage == {"prompt": 10, "completion": 5, "total": 15}


def test_token_usage_falls_back_to_the_message_metadata():
    usage = _token_usage(_result(None, {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}))

    assert usage == {"prompt": 7, "completion": 3, "total": 10}


def test_token_usage_is_zero_when_nothing_reports_it():
    assert _token_usage(_result()) == {"prompt": 0, "completion": 0, "total": 0}


def test_the_handler_logs_one_line_per_call_with_node_model_and_tokens(caplog):
    handler = LoggingCallbackHandler()
    run_id = uuid4()

    with caplog.at_level(logging.INFO, logger="rpids.llm"):
        handler.on_chat_model_start({}, [], run_id=run_id, metadata={"node": "planner"})
        handler.on_llm_end(
            _result(
                {
                    "model_name": "gpt-4o-mini",
                    "token_usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                }
            ),
            run_id=run_id,
        )

    [line] = [r.getMessage() for r in caplog.records if r.getMessage().startswith("llm_call")]
    assert "node=planner" in line
    assert "model=gpt-4o-mini" in line
    assert "tokens[p/c/t]=1/2/3" in line
    assert handler._runs == {}, "per-run bookkeeping must be released"


def test_an_explicit_run_name_beats_the_node_metadata(caplog):
    handler = LoggingCallbackHandler()
    run_id = uuid4()

    with caplog.at_level(logging.INFO, logger="rpids.llm"):
        handler.on_llm_start({}, ["p"], run_id=run_id, metadata={"node": "planner"}, name="thread_summary")
        handler.on_llm_end(_result(), run_id=run_id)

    assert "node=thread_summary" in caplog.text


def test_errors_are_logged_as_warnings_and_release_the_run(caplog):
    handler = LoggingCallbackHandler()
    run_id = uuid4()

    with caplog.at_level(logging.WARNING, logger="rpids.llm"):
        handler.on_chat_model_start({}, [], run_id=run_id, metadata={"node": "rag_path"})
        handler.on_llm_error(TimeoutError("deadline"), run_id=run_id)

    assert "llm_error node=rag_path error=deadline" in caplog.text
    assert handler._runs == {}


def test_an_end_without_a_start_still_logs_instead_of_raising(caplog):
    handler = LoggingCallbackHandler()

    with caplog.at_level(logging.INFO, logger="rpids.llm"):
        handler.on_llm_end(_result(), run_id=uuid4())

    assert "node=llm" in caplog.text
    assert "elapsed_ms=None" in caplog.text
