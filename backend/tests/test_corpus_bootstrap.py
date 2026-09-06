from unittest.mock import patch

import pytest

from src.ingestion import bootstrap
from src.ingestion.pipeline import IngestionReport, IngestionResult


def _report(*results: IngestionResult) -> IngestionReport:
    return IngestionReport(results=list(results))


def _indexed(name: str, nodes: int = 4) -> IngestionResult:
    return IngestionResult(file_name=name, doc_type="retention_policy", version="1.0", text_nodes=nodes)


def _skipped(name: str, reason: str) -> IngestionResult:
    return IngestionResult(file_name=name, skipped=True, reason=reason)


def test_a_stray_row_no_longer_suppresses_the_whole_corpus(tmp_path):
    with patch.object(bootstrap, "corpus_dir", return_value=tmp_path), \
         patch.object(bootstrap, "check_embed_model_compatibility", return_value=(True, "")), \
         patch.object(bootstrap, "table_has_rows", return_value=True), \
         patch.object(bootstrap, "unindexed_files", side_effect=[["Data_Retention.pdf"], []]), \
         patch.object(bootstrap, "ingest_directory", return_value=_report(_indexed("Data_Retention.pdf"))) as ingest:
        report = bootstrap.bootstrap_corpus()

    assert ingest.called
    assert report is not None
    assert report.total_nodes == 4


def test_a_complete_corpus_is_not_re_ingested(tmp_path):
    with patch.object(bootstrap, "corpus_dir", return_value=tmp_path), \
         patch.object(bootstrap, "check_embed_model_compatibility", return_value=(True, "")), \
         patch.object(bootstrap, "table_has_rows", return_value=True), \
         patch.object(bootstrap, "unindexed_files", return_value=[]), \
         patch.object(bootstrap, "ingest_directory") as ingest:
        report = bootstrap.bootstrap_corpus()

    assert report is None
    assert not ingest.called


def test_an_empty_table_ingests_everything(tmp_path):
    with patch.object(bootstrap, "corpus_dir", return_value=tmp_path), \
         patch.object(bootstrap, "check_embed_model_compatibility", return_value=(True, "")), \
         patch.object(bootstrap, "table_has_rows", return_value=False), \
         patch.object(bootstrap, "unindexed_files", return_value=[]), \
         patch.object(bootstrap, "ingest_directory", return_value=_report(_indexed("a.pdf"), _indexed("b.pdf"))) as ingest:
        report = bootstrap.bootstrap_corpus()

    assert ingest.called
    assert len(report.results) == 2


def test_a_document_still_missing_after_bootstrap_is_reported(tmp_path, caplog):
    with patch.object(bootstrap, "corpus_dir", return_value=tmp_path), \
         patch.object(bootstrap, "check_embed_model_compatibility", return_value=(True, "")), \
         patch.object(bootstrap, "table_has_rows", return_value=True), \
         patch.object(bootstrap, "unindexed_files", side_effect=[["Privacy.pdf"], ["Privacy.pdf"]]), \
         patch.object(bootstrap, "ingest_directory", return_value=_report(_skipped("Privacy.pdf", "parsing produced no indexable nodes"))), \
         caplog.at_level("ERROR"):
        bootstrap.bootstrap_corpus()

    assert "Privacy.pdf" in caplog.text
    assert "escalate on low confidence" in caplog.text


def test_an_incompatible_embedding_model_stops_the_bootstrap(tmp_path):
    with patch.object(bootstrap, "corpus_dir", return_value=tmp_path), \
         patch.object(bootstrap, "check_embed_model_compatibility", return_value=(False, "mismatch")), \
         pytest.raises(bootstrap.CorpusBootstrapError):
        bootstrap.bootstrap_corpus()


def test_an_empty_corpus_directory_is_an_error(tmp_path):
    missing = tmp_path / "not_here"

    with patch.object(bootstrap, "corpus_dir", return_value=missing), \
         patch.object(bootstrap, "check_embed_model_compatibility", return_value=(True, "")), \
         pytest.raises(bootstrap.CorpusBootstrapError):
        bootstrap.bootstrap_corpus()


def test_readme_is_not_treated_as_a_corpus_document(tmp_path):
    (tmp_path / "README.md").write_text("notes")
    (tmp_path / "Retention_Policy.pdf").write_bytes(b"%PDF-1.4")

    with patch.object(bootstrap, "corpus_dir", return_value=tmp_path):
        names = [path.name for path in bootstrap.corpus_files()]

    assert names == ["Retention_Policy.pdf"]


def test_unindexed_files_names_what_the_hash_check_cannot_find(tmp_path):
    (tmp_path / "Retention_Policy.pdf").write_bytes(b"%PDF-1.4 retention")
    (tmp_path / "Privacy_Policy.pdf").write_bytes(b"%PDF-1.4 privacy")

    with patch.object(bootstrap, "corpus_dir", return_value=tmp_path), \
         patch.object(bootstrap, "file_hash", side_effect=lambda p: p), \
         patch.object(bootstrap, "file_hash_exists", side_effect=lambda h: "Retention" in h):
        missing = bootstrap.unindexed_files()

    assert missing == ["Privacy_Policy.pdf"]
