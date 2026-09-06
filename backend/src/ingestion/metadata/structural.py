import hashlib
import re
from datetime import UTC, date, datetime
from pathlib import Path

from src.ingestion.metadata.schema import (
    ALLOWED_DEPARTMENTS,
    ALLOWED_DOC_TYPES,
    DOMAIN,
    LEVEL_DOCUMENT,
)

FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)

FILENAME_HINTS = {
    "privacy": "privacy_policy",
    "retention": "retention_policy",
    "vendor": "vendor_policy",
    "supplier": "vendor_policy",
    "bribery": "anti_bribery_policy",
    "infosec": "infosec_policy",
    "security": "infosec_policy",
    "gdpr": "gdpr",
    "iso": "iso_27001",
}

DEFAULT_EFFECTIVE_DATE = date(2025, 1, 1)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    match = FRONTMATTER.match(text or "")

    if not match:
        return {}, text or ""

    values = {}

    for line in match.group(1).splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("\"'")

    return values, text[match.end() :]


def file_hash(file_path: str) -> str:
    digest = hashlib.sha256()

    with Path(file_path).open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)

    return digest.hexdigest()[:32]


def infer_doc_type(original_filename: str) -> str:
    lowered = (original_filename or "").lower()

    for hint, doc_type in FILENAME_HINTS.items():
        if hint in lowered:
            return doc_type

    return ""


def declared_doc_type(frontmatter: dict) -> str:
    declared = frontmatter.get("doc_type")

    return declared if declared in ALLOWED_DOC_TYPES else ""


def resolve_effective_date(frontmatter: dict) -> str:
    declared = frontmatter.get("effective_date", "")

    try:
        return date.fromisoformat(declared).isoformat()
    except ValueError:
        return DEFAULT_EFFECTIVE_DATE.isoformat()


def resolve_title(frontmatter: dict, original_filename: str) -> str:
    declared = frontmatter.get("title")

    if declared:
        return declared

    return Path(original_filename or "document").stem.replace("_", " ").title()


def build_structural_metadata(
    file_path: str,
    original_filename: str,
    parsed_with: str,
    parse_reason: str,
    frontmatter: dict,
) -> dict:
    metadata = {
        "domain": DOMAIN,
        "doc_type": declared_doc_type(frontmatter) or infer_doc_type(original_filename),
        "document_title": resolve_title(frontmatter, original_filename),
        "version": frontmatter.get("version", "1.0"),
        "effective_date": resolve_effective_date(frontmatter),
        "is_current": True,
        "original_file_name": original_filename,
        "file_name": Path(file_path).name,
        "file_type": Path(file_path).suffix.replace(".", ""),
        "file_size_kb": round(Path(file_path).stat().st_size / 1024, 1),
        "file_hash": file_hash(file_path),
        "ingested_at": datetime.now(UTC).isoformat(),
        "parsed_with": parsed_with,
        "parse_reason": parse_reason,
        "embed_model": "",
        "level": LEVEL_DOCUMENT,
    }

    declared_department = frontmatter.get("department")

    if declared_department in ALLOWED_DEPARTMENTS:
        metadata["owning_department"] = declared_department

    return metadata
