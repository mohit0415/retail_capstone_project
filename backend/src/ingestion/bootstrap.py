import logging
from pathlib import Path

from configs.settings import settings
from src.index.vector_index import (
    check_embed_model_compatibility,
    file_hash_exists,
    table_has_rows,
)
from src.ingestion.metadata.structural import file_hash
from src.ingestion.pipeline import IngestionReport, ingest_directory

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[2]

INGESTABLE_SUFFIXES = (".md", ".pdf", ".docx", ".txt")


class CorpusBootstrapError(RuntimeError):
    pass


def corpus_dir() -> Path:
    configured = Path(settings.policy_corpus_dir)

    if configured.is_absolute():
        return configured

    return BACKEND_ROOT / configured


def corpus_is_indexed() -> bool:
    return table_has_rows()


def corpus_files() -> list[Path]:
    directory = corpus_dir()

    if not directory.exists():
        return []

    found = []

    for pattern in INGESTABLE_SUFFIXES:
        found.extend(sorted(directory.glob(f"*{pattern}")))

    return [path for path in found if path.name.lower() != "readme.md"]


def unindexed_files() -> list[str]:
    missing = []

    for path in corpus_files():
        try:
            present = file_hash_exists(file_hash(str(path)))
        except Exception as exc:
            logger.warning("could not check whether %s is indexed: %s", path.name, exc)
            continue

        if not present:
            missing.append(path.name)

    return missing


def bootstrap_corpus() -> IngestionReport | None:
    if not settings.bootstrap_corpus_on_startup:
        logger.info("corpus bootstrap disabled by configuration")
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

    if table_has_rows():
        missing = unindexed_files()

        if not missing:
            logger.info("every document in %s is already indexed", directory)

            return None

        logger.warning(
            "the vector table holds rows but %s of the corpus is not indexed (%s); "
            "reconciling rather than trusting the table to be complete",
            len(missing),
            ", ".join(missing),
        )
    else:
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

    if not indexed and not table_has_rows():
        raise CorpusBootstrapError(
            f"every document in {directory} was skipped, so the knowledge base is still empty. "
            f"First reason: {failed[0].reason if failed else 'unknown'}"
        )

    still_missing = unindexed_files()

    if still_missing:
        logger.error(
            "these corpus documents are still not indexed after bootstrap, so questions they "
            "answer will retrieve nothing and escalate on low confidence: %s",
            ", ".join(still_missing),
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
