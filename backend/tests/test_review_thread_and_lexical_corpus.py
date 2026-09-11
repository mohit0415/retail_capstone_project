"""Fixes found in the running server's log (README §18 follow-up):

* the review package showed ``conversation_history: []`` for every first-turn
  request, because the state only carries *prior* turns and the current one is
  written after the response;
* ``request_latency`` was missing from a database initialised before it was
  added, so every request logged a write failure and ``/metrics/slo`` was 503;
* the BM25 leg of hybrid retrieval never ran after a restart, because
  ``VectorStoreIndex.from_vector_store`` leaves the docstore empty;
* generated NL2SQL joined tables with a bare ``vendor_id`` and failed as ambiguous.
"""

import json
from pathlib import Path

from llama_index.core.schema import TextNode

from src.index import vector_index
from src.nodes.escalation import build_context_package, thread_for_reviewer
from src.prompts.library import NL2SQL_GENERATION
from src.retrieval import fusion
from tests.conftest import base_state

ROOT = Path(__file__).resolve().parents[1]


def test_a_first_question_appears_in_the_review_thread():
    state = base_state("How long must we retain customer personal data?")

    thread = thread_for_reviewer(state)

    assert len(thread) == 1
    assert thread[0]["role"] == "user"
    assert thread[0]["content"] == "How long must we retain customer personal data?"
    assert thread[0]["current"] is True
    assert thread[0]["request_id"] == "req-test"


def test_prior_turns_come_first_and_the_current_question_last():
    state = base_state(
        "And for supplier invoices?",
        conversation_history=[
            {"role": "user", "content": "How long must we retain customer personal data?"},
            {"role": "assistant", "content": "Seven years [Retention Policy §4.2]."},
        ],
    )

    thread = thread_for_reviewer(state)

    assert [turn["role"] for turn in thread] == ["user", "assistant", "user"]
    assert thread[-1]["content"] == "And for supplier invoices?"
    assert "current" not in thread[0]


def test_the_context_package_is_never_empty_on_history_and_counts_prior_turns():
    state = base_state("May I accept this gift?")

    package = build_context_package(state)

    assert package["conversation_history"][-1]["content"] == "May I accept this gift?"
    assert package["prior_turns"] == 0
    assert package["original_query"] == "May I accept this gift?"


def test_the_graph_state_itself_is_not_changed_by_the_package():
    state = base_state("May I accept this gift?")

    build_context_package(state)

    assert state["conversation_history"] == []


def test_startup_ddl_creates_every_table_the_request_path_writes_to():
    ddl = (ROOT / "sql" / "conversation_schema.sql").read_text()

    for table in (
        "conversation_threads",
        "conversation_turns",
        "system_audit_log",
        "escalation_queue",
        "request_latency",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in ddl, table

    assert "request_latency_request_idx" in ddl
    assert ddl.count("IF NOT EXISTS") >= 12


def test_startup_ddl_matches_the_install_schema_for_request_latency():
    startup = (ROOT / "sql" / "conversation_schema.sql").read_text()
    install = (ROOT / "data" / "sql" / "schema.sql").read_text()

    def _columns(text: str) -> list[str]:
        start = text.index("CREATE TABLE IF NOT EXISTS request_latency")
        body = text[start : text.index(");", start)]

        return sorted(line.split()[0] for line in body.splitlines()[1:] if line.strip())

    assert _columns(startup) == _columns(install)


def test_generated_sql_prompt_requires_qualified_columns_on_joins():
    assert "alias every table" in NL2SQL_GENERATION
    assert "qualify every column" in NL2SQL_GENERATION
    assert "ambiguous" in NL2SQL_GENERATION


def _node(text: str, doc_type: str = "retention_policy", is_current: bool = True) -> TextNode:
    return TextNode(
        id_=f"n-{abs(hash(text))}",
        text=text,
        metadata={
            "doc_type": doc_type,
            "document_title": "Retention Policy",
            "clause_number": "4.2",
            "is_current": is_current,
        },
    )


class _EmptyDocstore:
    def __init__(self) -> None:
        self.docs: dict = {}


class _Index:
    def __init__(self) -> None:
        self.docstore = _EmptyDocstore()


def test_bm25_leg_falls_back_to_the_vector_table_when_the_docstore_is_empty(monkeypatch):
    corpus = [
        _node("Customer invoices are retained for seven years after the transaction date."),
        _node("Vendors are reviewed annually.", doc_type="vendor_policy"),
        _node("Superseded clause about invoices.", is_current=False),
    ]
    monkeypatch.setattr(fusion, "load_lexical_corpus", lambda: corpus)

    retriever = fusion._build_bm25_retriever(_Index(), ["retention_policy"], top_k=5)

    assert retriever is not None

    hits = retriever.retrieve("how long are invoices retained")

    assert hits
    assert hits[0].node.metadata["doc_type"] == "retention_policy"
    assert all(hit.node.metadata.get("is_current") is not False for hit in hits)


def test_bm25_leg_is_skipped_when_nothing_is_in_scope(monkeypatch):
    monkeypatch.setattr(fusion, "load_lexical_corpus", lambda: [_node("x", doc_type="vendor_policy")])

    assert fusion._build_bm25_retriever(_Index(), ["gdpr"], top_k=5) is None


def test_rows_are_rebuilt_from_the_llama_index_envelope_or_plain_metadata():
    envelope = {
        "_node_content": json.dumps(
            {
                "id_": "node-1",
                "text": "",
                "metadata": {"doc_type": "gdpr", "clause_number": "17"},
                "excluded_embed_metadata_keys": [],
                "excluded_llm_metadata_keys": [],
                "relationships": {},
                "class_name": "TextNode",
            }
        ),
        "_node_type": "TextNode",
        "doc_type": "gdpr",
        "clause_number": "17",
    }

    node = vector_index._row_to_node("node-1", "Right to erasure.", envelope)

    assert node.node_id == "node-1"
    assert node.get_content() == "Right to erasure."
    assert node.metadata["clause_number"] == "17"

    plain = vector_index._row_to_node(
        "node-2", "Plain text.", json.dumps({"doc_type": "iso_27001", "_private": 1})
    )

    assert plain.node_id == "node-2"
    assert plain.metadata == {"doc_type": "iso_27001"}


def test_the_lexical_corpus_is_cached_and_dropped_on_reset(monkeypatch):
    calls = {"n": 0}

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, statement):
            calls["n"] += 1

            class _Result:
                @staticmethod
                def fetchall():
                    return [("id-1", "Seven years.", {"doc_type": "retention_policy"})]

            return _Result()

    class _Engine:
        def connect(self):
            return _Conn()

    monkeypatch.setattr(vector_index, "create_engine", lambda url: _Engine())
    monkeypatch.setattr(vector_index, "_lexical_corpus", None)

    first = vector_index.load_lexical_corpus()
    second = vector_index.load_lexical_corpus()

    assert len(first) == 1 and first is second
    assert calls["n"] == 1

    vector_index.reset_index()
    vector_index.load_lexical_corpus()

    assert calls["n"] == 2


def test_an_unreadable_table_yields_an_empty_corpus_not_an_error(monkeypatch):
    def _boom(url):
        raise RuntimeError("no database")

    monkeypatch.setattr(vector_index, "create_engine", _boom)
    monkeypatch.setattr(vector_index, "_lexical_corpus", None)

    assert vector_index.load_lexical_corpus() == []
