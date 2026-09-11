"""``src/core/handoff.py`` (explicit human hand-off + escalation e-mail) and
``src/core/conversation.py`` (thread history and summaries).

The database and SMTP are replaced with in-memory fakes, so every branch runs
without infrastructure: skipped when disabled, skipped when mis-configured,
sent, and failed.
"""

import datetime
from contextlib import contextmanager
from typing import ClassVar

import pytest

from configs.settings import settings
from src.core import conversation, handoff


@pytest.mark.parametrize(
    "question",
    [
        "I want to speak to a human about this",
        "please escalate this to legal review",
        "can I talk to a compliance officer?",
        "Contact Support for me",
    ],
)
def test_explicit_requests_for_a_person_are_recognised(question):
    assert handoff.requested_human(question) is True


@pytest.mark.parametrize(
    "question",
    ["what is the retention period for invoices", "is vendor Acme approved", "", None],
)
def test_ordinary_questions_are_not_a_handoff(question):
    assert handoff.requested_human(question) is False


def test_reference_ids_are_timestamped_and_unique():
    at = datetime.datetime(2026, 9, 7, 10, 30, 15, tzinfo=datetime.UTC)

    first = handoff.generate_reference_id(at)
    second = handoff.generate_reference_id(at)

    assert first.startswith("ESC-20260907-103015-")
    assert len(first) == len("ESC-20260907-103015-") + 6
    assert first != second


def test_a_freshly_generated_reference_uses_now():
    assert handoff.generate_reference_id().startswith(f"ESC-{datetime.datetime.now(datetime.UTC):%Y%m%d}-")


class FakeSMTP:
    instances: ClassVar[list["FakeSMTP"]] = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.tls = False
        self.login_args = None
        self.sent = []
        self.fail_login = False
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        self.tls = True

    def login(self, username, password):
        if self.fail_login:
            raise RuntimeError("535 authentication failed")

        self.login_args = (username, password)

    def send_message(self, message):
        self.sent.append(message)


@pytest.fixture
def smtp(monkeypatch):
    FakeSMTP.instances.clear()
    monkeypatch.setattr(handoff.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(settings, "escalation_email_enabled", True)
    monkeypatch.setattr(settings, "smtp_host", "smtp.example.com")
    monkeypatch.setattr(settings, "smtp_port", 2525)
    monkeypatch.setattr(settings, "smtp_username", "bot")
    monkeypatch.setattr(settings, "smtp_password", "secret")
    monkeypatch.setattr(settings, "application_email", "bot@example.com")
    monkeypatch.setattr(settings, "reviewer_email", "reviewer@example.com")

    return FakeSMTP


def _context(**overrides) -> dict:
    context = {
        "reference_id": "ESC-1",
        "request_id": "req-1",
        "thread_id": "thread-1",
        "user_id": "asker",
        "role": "store_manager",
        "risk_level": "High",
        "confidence": 0.42,
        "evidence_path": "rag",
        "reason": "risk level is High",
        "standalone_query": "may I accept a gift from a vendor?",
        "draft_answer": "No [Anti Bribery Policy §3.1].",
        "validation_output": {"passed": False},
        "citations": ["Anti Bribery Policy §3.1"],
        "sql_evidence": None,
        "reasoning_trace": [{"node": "planner"}],
    }
    context.update(overrides)

    return context


def test_the_email_is_skipped_when_the_feature_is_off(smtp, monkeypatch):
    monkeypatch.setattr(settings, "escalation_email_enabled", False)

    assert handoff.send_escalation_email(_context()) is False
    assert smtp.instances == []


def test_the_email_is_skipped_and_the_gap_named_when_smtp_is_incomplete(smtp, monkeypatch, caplog):
    monkeypatch.setattr(settings, "smtp_password", "")
    monkeypatch.setattr(settings, "reviewer_email", "")

    with caplog.at_level("WARNING", logger="src.core.handoff"):
        sent = handoff.send_escalation_email(_context())

    assert sent is False
    assert smtp.instances == []
    assert "SMTP_PASSWORD" in caplog.text
    assert "REVIEWER_EMAIL" in caplog.text
    assert "still queued" in caplog.text


def test_a_complete_configuration_sends_one_message_with_the_package(smtp):
    assert handoff.send_escalation_email(_context()) is True

    [server] = smtp.instances
    assert (server.host, server.port, server.timeout) == ("smtp.example.com", 2525, 15)
    assert server.tls is True
    assert server.login_args == ("bot", "secret")

    [message] = server.sent
    assert message["From"] == "bot@example.com"
    assert message["To"] == "reviewer@example.com"
    assert message["Subject"] == "[COMPLIANCE ESCALATION] ESC-1 - risk level is High"

    body = message.get_payload(decode=True).decode("utf-8")
    assert "may I accept a gift from a vendor?" in body
    assert "No [Anti Bribery Policy §3.1]." in body
    assert '"passed": false' in body
    assert f"{settings.reviewer_console_url}/req-1" in body


def test_an_smtp_failure_is_logged_and_reported_as_not_sent(smtp, caplog):
    def _failing(host, port, timeout=None):
        server = FakeSMTP(host, port, timeout)
        server.fail_login = True

        return server

    handoff.smtplib.SMTP = _failing

    with caplog.at_level("ERROR", logger="src.core.handoff"):
        sent = handoff.send_escalation_email(_context())

    assert sent is False
    assert "escalation e-mail failed" in caplog.text
    assert "ESC-1" in caplog.text


class FakeConnection:
    """Records every statement and answers from a scripted queue."""

    def __init__(self, answers: dict):
        self.answers = answers
        self.executed: list[tuple[str, dict | None]] = []

    def execute(self, sql, params=None):
        self.executed.append((sql.strip().split("\n")[0], params))

        for needle, rows in self.answers.items():
            if needle in sql:
                return _Cursor(rows)

        return _Cursor([])


class _Cursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


@pytest.fixture
def db(monkeypatch):
    def _install(answers: dict | None = None, broken: bool = False):
        conn = FakeConnection(answers or {})

        @contextmanager
        def _connection():
            if broken:
                raise ConnectionError("database down")

            yield conn

        monkeypatch.setattr(conversation, "read_only_connection", _connection)
        monkeypatch.setattr(conversation, "writable_connection", _connection)

        return conn

    return _install


def test_history_comes_back_oldest_first_as_role_content_pairs(db):
    db(
        {
            "FROM conversation_turns": [
                {"speaker": "assistant", "content": "seven years", "created_at": 2},
                {"speaker": "user", "content": "how long are invoices kept", "created_at": 1},
            ]
        }
    )

    assert conversation.load_history("t1") == [
        {"role": "user", "content": "how long are invoices kept"},
        {"role": "assistant", "content": "seven years"},
    ]


def test_history_honours_the_configured_window(db):
    conn = db()

    conversation.load_history("t1")
    conversation.load_history("t1", limit=3)

    assert conn.executed[0][1]["limit"] == settings.conversation_window_turns
    assert conn.executed[1][1]["limit"] == 3


def test_a_database_outage_degrades_to_an_empty_thread(db, caplog):
    db(broken=True)

    with caplog.at_level("WARNING", logger="src.core.conversation"):
        assert conversation.load_thread("t1") == ([], "")

    assert "history unavailable" in caplog.text
    assert "summary unavailable" in caplog.text


def test_a_missing_or_null_summary_is_an_empty_string(db):
    db({"FROM conversation_threads": []})
    assert conversation.load_summary("t1") == ""

    db({"FROM conversation_threads": [{"summary": None}]})
    assert conversation.load_summary("t1") == ""

    db({"FROM conversation_threads": [{"summary": "asked about invoices"}]})
    assert conversation.load_summary("t1") == "asked about invoices"


def test_empty_content_is_never_written(db):
    conn = db()

    conversation.append_turn("t1", "u1", "r1", "assistant", "")

    assert conn.executed == []


def test_record_exchange_writes_both_sides_then_checks_the_turn_count(db):
    conn = db({"COUNT(*) AS turns": [{"turns": 2}]})

    conversation.record_exchange("t1", "u1", "r1", "how long?", "seven years")

    inserts = [params for sql, params in conn.executed if sql.startswith("INSERT INTO conversation_turns")]
    assert [p["speaker"] for p in inserts] == ["user", "assistant"]
    assert inserts[0]["content"] == "how long?"
    assert inserts[1]["request_id"] == "r1"

    assert not any("conversation_threads" in sql for sql, _ in conn.executed)


def test_an_append_failure_is_logged_not_raised(db, caplog):
    db(broken=True)

    with caplog.at_level("ERROR", logger="src.core.conversation"):
        conversation.append_turn("t1", "u1", "r1", "user", "hello")

    assert "could not append user turn" in caplog.text


@pytest.fixture
def summariser(monkeypatch):
    calls = []

    class _Response:
        content = "  The user asked about invoice retention; the answer was seven years.  "

    class _Model:
        def invoke(self, messages):
            calls.append(messages)

            return _Response()

    monkeypatch.setattr(conversation, "model_for", lambda name: _Model())
    monkeypatch.setattr(
        conversation, "render_prompt", lambda name, fallback, **kw: f"SUMMARISE:{kw['transcript']}"
    )

    return calls


def test_the_summary_is_refreshed_once_the_thread_is_long_enough(db, summariser, monkeypatch):
    monkeypatch.setattr(settings, "conversation_summary_after_turns", 4)
    conn = db(
        {
            "COUNT(*) AS turns": [{"turns": 4}],
            "FROM conversation_turns": [
                {"speaker": "assistant", "content": "seven years", "created_at": 2},
                {"speaker": "user", "content": "how long?", "created_at": 1},
            ],
        }
    )

    conversation.refresh_summary("t1")

    assert len(summariser) == 1
    [messages] = summariser
    assert messages[0].content == "SUMMARISE:user: how long?\nassistant: seven years"

    upsert = next(params for sql, params in conn.executed if "conversation_threads" in sql)
    assert upsert == {
        "thread_id": "t1",
        "summary": "The user asked about invoice retention; the answer was seven years.",
        "turn_count": 4,
    }


def test_no_summary_is_generated_below_the_threshold(db, summariser, monkeypatch):
    monkeypatch.setattr(settings, "conversation_summary_after_turns", 8)
    db({"COUNT(*) AS turns": [{"turns": 7}]})

    conversation.refresh_summary("t1")

    assert summariser == []


def test_a_summariser_failure_leaves_the_old_summary_in_place(db, monkeypatch, caplog):
    monkeypatch.setattr(settings, "conversation_summary_after_turns", 1)
    conn = db(
        {
            "COUNT(*) AS turns": [{"turns": 3}],
            "FROM conversation_turns": [{"speaker": "user", "content": "hi", "created_at": 1}],
        }
    )

    class _Model:
        def invoke(self, messages):
            raise TimeoutError("model timed out")

    monkeypatch.setattr(conversation, "model_for", lambda name: _Model())
    monkeypatch.setattr(conversation, "render_prompt", lambda name, fallback, **kw: "p")

    with caplog.at_level("WARNING", logger="src.core.conversation"):
        conversation.refresh_summary("t1")

    assert "summary generation failed" in caplog.text
    assert not any("conversation_threads" in sql for sql, _ in conn.executed)


def test_a_turn_count_failure_skips_the_summary_quietly(db, summariser, caplog):
    db(broken=True)

    with caplog.at_level("WARNING", logger="src.core.conversation"):
        conversation.refresh_summary("t1")

    assert summariser == []
    assert "could not count turns" in caplog.text
