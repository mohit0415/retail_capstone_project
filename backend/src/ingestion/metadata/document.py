import logging

from src.ingestion.elements import ParsedDocument
from src.ingestion.metadata.content import extract_content_metadata
from src.ingestion.metadata.schema import ALLOWED_DOC_TYPES, validate_document_metadata
from src.ingestion.metadata.structural import build_structural_metadata

logger = logging.getLogger(__name__)


class MetadataContractError(ValueError):
    pass


def build_document_metadata(
    file_path: str,
    original_filename: str,
    parsed: ParsedDocument,
    frontmatter: dict,
    body: str,
    embed_model: str,
) -> dict:
    metadata = build_structural_metadata(
        file_path=file_path,
        original_filename=original_filename,
        parsed_with=parsed.route.parser,
        parse_reason=parsed.route.reason,
        frontmatter=frontmatter,
    )

    inferred = extract_content_metadata(body)
    fallback_doc_type = inferred.pop("inferred_doc_type", "")

    for key, value in inferred.items():
        if key not in metadata:
            metadata[key] = value

    if not metadata["doc_type"]:
        metadata["doc_type"] = fallback_doc_type

    if metadata["doc_type"] not in ALLOWED_DOC_TYPES:
        raise MetadataContractError(
            f"{original_filename} could not be assigned a doc_type. doc_type is the key every "
            f"role scope filters on, so guessing it would either hide the document from everyone "
            f"or expose it to a role that should not see it. Add 'doc_type: <one of "
            f"{ALLOWED_DOC_TYPES}>' to the document frontmatter, or rename the file so it carries "
            f"a recognisable word such as privacy, retention, vendor, bribery, security, gdpr or iso."
        )

    metadata["embed_model"] = embed_model

    problems = validate_document_metadata(metadata)

    if problems:
        raise MetadataContractError(
            f"document metadata for {original_filename} is incomplete: {'; '.join(problems)}"
        )

    return metadata
