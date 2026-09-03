import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from llama_index.core.schema import BaseNode

from src.index.models import active_embed_model_name, configure_llama_settings
from src.index.vector_index import (
    check_embed_model_compatibility,
    file_hash_exists,
    insert_nodes,
    supersede_previous_versions,
)
from src.ingestion.image_nodes import build_image_nodes, extract_images_from_pdf
from src.ingestion.parser import parse_document
from src.ingestion.policy_metadata import build_document_metadata, file_hash, parse_frontmatter
from src.ingestion.splitters import ClauseAwareSplitter
from src.ingestion.table_nodes import build_table_nodes

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    file_name: str
    parsed_with: str = ""
    doc_type: str = ""
    version: str = ""
    text_nodes: int = 0
    table_nodes: int = 0
    image_nodes: int = 0
    superseded: int = 0
    skipped: bool = False
    reason: str = ""

    @property
    def total_nodes(self) -> int:
        return self.text_nodes + self.table_nodes + self.image_nodes


@dataclass
class IngestionReport:
    results: List[IngestionResult] = field(default_factory=list)

    @property
    def total_nodes(self) -> int:
        return sum(result.total_nodes for result in self.results)


def build_nodes(file_path: str, original_filename: str) -> tuple[List[BaseNode], dict, str]:
    raw_markdown, parsed_with = parse_document(file_path)
    frontmatter, body = parse_frontmatter(raw_markdown)

    document_metadata = build_document_metadata(
        file_path=file_path,
        original_filename=original_filename,
        parsed_with=parsed_with,
        frontmatter=frontmatter,
        body=body,
        embed_model=active_embed_model_name(),
    )

    splitter = ClauseAwareSplitter()

    nodes: List[BaseNode] = splitter.split_document(body, document_metadata)

    nodes.extend(build_table_nodes(body, document_metadata))

    if Path(file_path).suffix.lower() == ".pdf":
        with tempfile.TemporaryDirectory() as workspace:
            images = extract_images_from_pdf(file_path, workspace)
            nodes.extend(build_image_nodes(images, document_metadata))

    return nodes, document_metadata, parsed_with


def ingest_file(file_path: str, original_filename: str | None = None, force: bool = False) -> IngestionResult:
    configure_llama_settings()

    name = original_filename or Path(file_path).name
    result = IngestionResult(file_name=name)

    if not force and file_hash_exists(file_hash(file_path)):
        result.skipped = True
        result.reason = "identical file content is already indexed"

        return result

    nodes, metadata, parsed_with = build_nodes(file_path, name)

    if not nodes:
        result.skipped = True
        result.reason = "parsing produced no indexable nodes"

        return result

    insert_nodes(nodes)

    result.parsed_with = parsed_with
    result.doc_type = metadata["doc_type"]
    result.version = metadata["version"]
    result.text_nodes = sum(1 for node in nodes if node.metadata.get("content_type") == "text")
    result.table_nodes = sum(1 for node in nodes if node.metadata.get("content_type") == "table_summary")
    result.image_nodes = sum(1 for node in nodes if node.metadata.get("content_type") == "image_caption")
    result.superseded = supersede_previous_versions(metadata["doc_type"], metadata["version"])

    return result


def ingest_directory(directory: str, force: bool = False) -> IngestionReport:
    configure_llama_settings()

    compatible, message = check_embed_model_compatibility()

    if not compatible:
        raise RuntimeError(message)

    report = IngestionReport()

    patterns = ("*.md", "*.pdf", "*.docx", "*.txt")
    paths: List[Path] = []

    for pattern in patterns:
        paths.extend(sorted(Path(directory).glob(pattern)))

    for path in paths:
        if path.name.lower() == "readme.md":
            continue

        try:
            report.results.append(ingest_file(str(path), path.name, force=force))
        except Exception as exc:
            logger.error("ingestion failed for %s: %s", path.name, exc)
            report.results.append(IngestionResult(file_name=path.name, skipped=True, reason=str(exc)))

    return report
