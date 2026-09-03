DOMAIN = "retail_compliance"

PARSER_LLAMAPARSE = "llamaparse"
PARSER_LLAMAINDEX = "llamaindex"

CONTENT_TYPE_TEXT = "text"
CONTENT_TYPE_TABLE = "table_summary"
CONTENT_TYPE_IMAGE = "image_caption"

MODALITY_TEXT = "text"
MODALITY_TABLE = "table"
MODALITY_DIAGRAM = "diagram"

LEVEL_DOCUMENT = "document"
LEVEL_NODE = "node"

UNATTRIBUTED_CLAUSE = ""

ALLOWED_DOC_TYPES = [
    "privacy_policy",
    "retention_policy",
    "vendor_policy",
    "anti_bribery_policy",
    "infosec_policy",
    "gdpr",
    "iso_27001",
]

ALLOWED_DEPARTMENTS = [
    "legal",
    "finance",
    "it",
    "hr",
    "marketing",
    "logistics",
    "facilities",
    "all",
]

ALLOWED_CONTENT_DOMAINS = ["obligation", "definition", "procedure", "reference", "mixed"]

ALLOWED_ROUTES = ["rag", "sql", "hybrid"]

ALLOWED_PARSERS = [PARSER_LLAMAPARSE, PARSER_LLAMAINDEX]

ALLOWED_CONTENT_TYPES = [CONTENT_TYPE_TEXT, CONTENT_TYPE_TABLE, CONTENT_TYPE_IMAGE]

ALLOWED_MODALITIES = [MODALITY_TEXT, MODALITY_TABLE, MODALITY_DIAGRAM]

MODALITY_OF = {
    CONTENT_TYPE_TEXT: MODALITY_TEXT,
    CONTENT_TYPE_TABLE: MODALITY_TABLE,
    CONTENT_TYPE_IMAGE: MODALITY_DIAGRAM,
}

DOCUMENT_KEYS = (
    "domain",
    "doc_type",
    "document_title",
    "version",
    "effective_date",
    "is_current",
    "owning_department",
    "topic",
    "keywords",
    "summary",
    "content_domain",
    "intended_route",
    "original_file_name",
    "file_name",
    "file_type",
    "file_size_kb",
    "file_hash",
    "ingested_at",
    "parsed_with",
    "parse_reason",
    "embed_model",
    "level",
)

NODE_KEYS = (
    "level",
    "content_type",
    "modality",
    "section",
    "clause_number",
    "citation",
    "element_label",
)

REQUIRED_NODE_KEYS = tuple(key for key in DOCUMENT_KEYS if key != "level") + NODE_KEYS

CLOSED_ENUMS = {
    "doc_type": ALLOWED_DOC_TYPES,
    "owning_department": ALLOWED_DEPARTMENTS,
    "content_domain": ALLOWED_CONTENT_DOMAINS,
    "intended_route": ALLOWED_ROUTES,
    "parsed_with": ALLOWED_PARSERS,
}

NODE_ENUMS = {
    "content_type": ALLOWED_CONTENT_TYPES,
    "modality": ALLOWED_MODALITIES,
    "level": [LEVEL_NODE],
}

EXCLUDED_METADATA_KEYS = [
    "original_file_name",
    "file_name",
    "file_type",
    "file_size_kb",
    "file_hash",
    "ingested_at",
    "parsed_with",
    "parse_reason",
    "embed_model",
    "level",
    "is_current",
    "element_label",
    "table_index",
    "image_index",
    "image_kind",
    "chunk_index",
    "page_label",
    "original_table",
    "image_path",
]


def validate_document_metadata(metadata: dict) -> list[str]:
    problems = []

    for key in DOCUMENT_KEYS:
        if key not in metadata:
            problems.append(f"missing document key '{key}'")

    for key, allowed in CLOSED_ENUMS.items():
        value = metadata.get(key)

        if value is not None and value not in allowed:
            problems.append(f"'{key}' is '{value}', which is outside {allowed}")

    if metadata.get("level") != LEVEL_DOCUMENT:
        problems.append(f"document metadata must carry level='{LEVEL_DOCUMENT}'")

    return problems


def validate_node_metadata(metadata: dict) -> list[str]:
    problems = []

    for key in REQUIRED_NODE_KEYS:
        if key not in metadata:
            problems.append(f"missing node key '{key}'")

    for key, allowed in {**CLOSED_ENUMS, **NODE_ENUMS}.items():
        value = metadata.get(key)

        if value is not None and value not in allowed:
            problems.append(f"'{key}' is '{value}', which is outside {allowed}")

    content_type = metadata.get("content_type")
    expected_modality = MODALITY_OF.get(content_type)

    if expected_modality and metadata.get("modality") != expected_modality:
        problems.append(
            f"content_type '{content_type}' must carry modality '{expected_modality}', "
            f"found '{metadata.get('modality')}'"
        )

    return problems


def assert_nodes_carry_contract(nodes) -> None:
    for position, node in enumerate(nodes):
        problems = validate_node_metadata(getattr(node, "metadata", None) or {})

        if problems:
            raise ValueError(
                f"node {position} does not satisfy the metadata contract and would be "
                f"invisible or wrongly visible to the scope filters: {'; '.join(problems)}"
            )
