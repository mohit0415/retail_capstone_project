from src.ingestion.elements import ClauseSection, RawImage, RawTable
from src.ingestion.metadata.schema import (
    CONTENT_TYPE_IMAGE,
    CONTENT_TYPE_TABLE,
    CONTENT_TYPE_TEXT,
    EXCLUDED_METADATA_KEYS,
    LEVEL_NODE,
    MODALITY_OF,
    UNATTRIBUTED_CLAUSE,
)

FIGURE_SECTION = "Figures"


def clause_citation(document_metadata: dict, clause_number: str) -> str:
    title = document_metadata.get("document_title", "Document")

    if not clause_number:
        return title

    return f"{title} §{clause_number}"


def page_citation(document_metadata: dict, page: int) -> str:
    title = document_metadata.get("document_title", "Document")

    return f"{title}, page {page}"


def _base_metadata(document_metadata: dict, content_type: str) -> dict:
    metadata = dict(document_metadata)

    metadata["level"] = LEVEL_NODE
    metadata["content_type"] = content_type
    metadata["modality"] = MODALITY_OF[content_type]

    return metadata


def text_node_metadata(document_metadata: dict, section: ClauseSection, chunk_index: int) -> dict:
    metadata = _base_metadata(document_metadata, CONTENT_TYPE_TEXT)

    metadata["section"] = section.heading
    metadata["clause_number"] = section.clause_number
    metadata["citation"] = clause_citation(document_metadata, section.clause_number)
    metadata["element_label"] = f"clause_{section.clause_number}_chunk_{chunk_index}"
    metadata["chunk_index"] = chunk_index

    return metadata


def table_node_metadata(document_metadata: dict, table: RawTable) -> dict:
    metadata = _base_metadata(document_metadata, CONTENT_TYPE_TABLE)

    metadata["section"] = table.heading
    metadata["clause_number"] = table.clause_number
    metadata["citation"] = clause_citation(document_metadata, table.clause_number)
    metadata["element_label"] = f"table_{table.table_index}"
    metadata["table_index"] = table.table_index
    metadata["original_table"] = table.markdown

    return metadata


def image_node_metadata(document_metadata: dict, image: RawImage, stored_path: str) -> dict:
    metadata = _base_metadata(document_metadata, CONTENT_TYPE_IMAGE)

    metadata["section"] = FIGURE_SECTION
    metadata["clause_number"] = UNATTRIBUTED_CLAUSE
    metadata["citation"] = page_citation(document_metadata, image.page)
    slot = "fig" if image.kind == "figure" else "img"

    metadata["element_label"] = f"page{image.page}_{slot}{image.image_index}"
    metadata["page"] = image.page
    metadata["page_label"] = str(image.page)
    metadata["image_index"] = image.image_index
    metadata["image_kind"] = image.kind
    metadata["image_path"] = stored_path

    return metadata


def apply_exclusions(node):
    node.excluded_embed_metadata_keys = list(EXCLUDED_METADATA_KEYS)
    node.excluded_llm_metadata_keys = list(EXCLUDED_METADATA_KEYS)

    return node
