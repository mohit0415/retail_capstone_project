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
from src.ingestion.elements import ParsedDocument
from src.ingestion.extractors.clause_extraction import extract_clauses
from src.ingestion.extractors.image_extraction import extract_images_from_pdf
from src.ingestion.extractors.table_extraction import extract_tables
from src.ingestion.loaders.dispatch import load_document
from src.ingestion.metadata.document import build_document_metadata
from src.ingestion.metadata.schema import (
    CONTENT_TYPE_IMAGE,
    CONTENT_TYPE_TABLE,
    CONTENT_TYPE_TEXT,
    assert_nodes_carry_contract,
)
from src.ingestion.metadata.structural import file_hash, parse_frontmatter
from src.ingestion.processors.imageprocessing import ImageProcessor
from src.ingestion.processors.tableprocessing import TableProcessor
from src.ingestion.processors.textprocessing import TextProcessor

logger = logging.getLogger(__name__)

INGESTABLE_PATTERNS = ("*.md", "*.pdf", "*.docx", "*.txt")


@dataclass
class IngestionResult:
    file_name: str
    parsed_with: str = ""
    parse_reason: str = ""
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


def build_media_nodes(file_path: str, parsed: ParsedDocument, sections, document_metadata) -> tuple:
    if not parsed.route.is_multimodal:
        logger.info(
            "no table or image node built: the document went through the text reader because %s",
            parsed.route.reason,
        )

        return [], [], sections

    tables, reduced = extract_tables(sections)
    table_nodes = TableProcessor().process(tables, document_metadata) if tables else []

    image_nodes = []

    if parsed.route.has_images and Path(file_path).suffix.lower() == ".pdf":
        with tempfile.TemporaryDirectory() as workspace:
            images = extract_images_from_pdf(file_path, workspace)

            if images:
                image_nodes = ImageProcessor().process(images, document_metadata)

    return table_nodes, image_nodes, reduced


def build_nodes(file_path: str, original_filename: str) -> tuple[List[BaseNode], dict, ParsedDocument]:
    parsed = load_document(file_path)

    frontmatter, body = parse_frontmatter(parsed.text)

    document_metadata = build_document_metadata(
        file_path=file_path,
        original_filename=original_filename,
        parsed=parsed,
        frontmatter=frontmatter,
        body=body,
        embed_model=active_embed_model_name(),
    )

    sections = extract_clauses(body)

    table_nodes, image_nodes, sections = build_media_nodes(
        file_path, parsed, sections, document_metadata
    )

    nodes: List[BaseNode] = TextProcessor().process(sections, document_metadata)
    nodes.extend(table_nodes)
    nodes.extend(image_nodes)

    assert_nodes_carry_contract(nodes)

    return nodes, document_metadata, parsed


def count_nodes(nodes: List[BaseNode], content_type: str) -> int:
    return sum(1 for node in nodes if node.metadata.get("content_type") == content_type)


def ingest_file(file_path: str, original_filename: str | None = None, force: bool = False) -> IngestionResult:
    configure_llama_settings()

    name = original_filename or Path(file_path).name
    result = IngestionResult(file_name=name)

    if not force and file_hash_exists(file_hash(file_path)):
        result.skipped = True
        result.reason = "identical file content is already indexed"

        return result

    nodes, metadata, parsed = build_nodes(file_path, name)

    if not nodes:
        result.skipped = True
        result.reason = "parsing produced no indexable nodes"

        return result

    insert_nodes(nodes)

    result.parsed_with = parsed.route.parser
    result.parse_reason = parsed.route.reason
    result.doc_type = metadata["doc_type"]
    result.version = metadata["version"]
    result.text_nodes = count_nodes(nodes, CONTENT_TYPE_TEXT)
    result.table_nodes = count_nodes(nodes, CONTENT_TYPE_TABLE)
    result.image_nodes = count_nodes(nodes, CONTENT_TYPE_IMAGE)
    result.superseded = supersede_previous_versions(metadata["doc_type"], metadata["version"])

    return result


def ingest_directory(directory: str, force: bool = False) -> IngestionReport:
    configure_llama_settings()

    compatible, message = check_embed_model_compatibility()

    if not compatible:
        raise RuntimeError(message)

    report = IngestionReport()

    paths: List[Path] = []

    for pattern in INGESTABLE_PATTERNS:
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
