import logging
from pathlib import Path

from configs.settings import settings
from src.index.vector_index import check_embed_model_compatibility, table_has_rows
from src.ingestion.pipeline import IngestionReport, ingest_directory

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class CorpusBootstrapError(RuntimeError):
    pass


def corpus_dir() -> Path:
    configured = Path(settings.policy_corpus_dir)

    if configured.is_absolute():
        return configured

    return BACKEND_ROOT / configured


def corpus_is_indexed() -> bool:
    return table_has_rows()


def bootstrap_corpus() -> IngestionReport | None:
    if not settings.bootstrap_corpus_on_startup:
        logger.info("corpus bootstrap disabled by configuration")
        return None

    if corpus_is_indexed():
        logger.info("policy corpus already indexed, skipping bootstrap")
        return None

    compatible, message = check_embed_model_compatibility()

    if not compatible:
        raise CorpusBootstrapError(message)

    directory = corpus_dir()

    if not directory.exists():
        raise CorpusBootstrapError(
            f"policy corpus directory does not exist: {directory}. "
            f"Set POLICY_CORPUS_DIR or create the folder and add the policy documents."
        )

    logger.info("vector table is empty, ingesting the policy corpus from %s", directory)

    report = ingest_directory(str(directory))

    if not report.results:
        raise CorpusBootstrapError(
            f"no ingestable documents found in {directory}. "
            f"Expected .md, .pdf, .docx or .txt files. See data/policies/README.md."
        )

    indexed = [result for result in report.results if not result.skipped]
    failed = [result for result in report.results if result.skipped]

    for result in failed:
        logger.warning("bootstrap skipped %s: %s", result.file_name, result.reason)

    if not indexed:
        raise CorpusBootstrapError(
            f"every document in {directory} was skipped, so the knowledge base is still empty. "
            f"First reason: {failed[0].reason if failed else 'unknown'}"
        )

    logger.info(
        "corpus bootstrap complete: %s node(s) from %s file(s), %s skipped",
        report.total_nodes,
        len(indexed),
        len(failed),
    )

    return report


def run_startup_bootstrap() -> None:
    try:
        bootstrap_corpus()
    except Exception as exc:
        if settings.bootstrap_fail_fast:
            raise

        logger.error(
            "corpus bootstrap failed, the application will start with an empty or partial "
            "knowledge base and policy questions will not be answerable: %s",
            exc,
        )
