import hashlib
import json
import logging
import re
from datetime import UTC, date, datetime
from pathlib import Path

from src.index.models import get_llm

logger = logging.getLogger(__name__)

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

DOMAIN = "retail_compliance"

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

CONTENT_PROMPT = """You are tagging a chunk of a retail compliance corpus made of five internal
policies (privacy, retention, vendor, anti-bribery, information security), GDPR articles and
ISO 27001 Annex A controls.

Read the excerpt and return JSON with exactly these fields:
- "doc_type": one of {doc_types}
- "topic": a short 1-3 word subject
- "keywords": 3-5 specific keywords taken from the text
- "owning_department": one of {departments}
- "content_domain": one of {content_domains} (obligation = states a requirement, definition =
  defines a term, procedure = describes steps, reference = a table or list of values)
- "intended_route": one of {routes} (rag for policy text, sql for facts that live in records,
  hybrid when a rule must be checked against records)
- "summary": one sentence

Return only valid JSON. No markdown fences, no preamble.

Excerpt:
{excerpt}"""


def parse_frontmatter(text: str) -> tuple[dict, str]:
    match = FRONTMATTER.match(text)

    if not match:
        return {}, text

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

    return "privacy_policy"


def build_structural_metadata(
    file_path: str,
    original_filename: str,
    parsed_with: str,
    frontmatter: dict,
) -> dict:
    doc_type = frontmatter.get("doc_type") or infer_doc_type(original_filename)

    if doc_type not in ALLOWED_DOC_TYPES:
        doc_type = infer_doc_type(original_filename)

    effective = frontmatter.get("effective_date", "2025-01-01")

    try:
        effective_date = date.fromisoformat(effective)
    except ValueError:
        effective_date = date(2025, 1, 1)

    title = frontmatter.get("title") or Path(original_filename).stem.replace("_", " ").title()

    return {
        "domain": DOMAIN,
        "doc_type": doc_type,
        "document_title": title,
        "version": frontmatter.get("version", "1.0"),
        "effective_date": effective_date.isoformat(),
        "is_current": True,
        "original_file_name": original_filename,
        "file_name": Path(file_path).name,
        "file_type": Path(file_path).suffix.replace(".", ""),
        "file_size_kb": round(Path(file_path).stat().st_size / 1024, 1),
        "file_hash": file_hash(file_path),
        "ingested_at": datetime.now(UTC).isoformat(),
        "parsed_with": parsed_with,
        "embed_model": "",
        "level": "document",
    }


def extract_content_metadata(excerpt: str) -> dict:
    defaults = {
        "topic": "unknown",
        "keywords": [],
        "owning_department": "all",
        "content_domain": "mixed",
        "intended_route": "rag",
        "summary": "",
    }

    prompt = CONTENT_PROMPT.format(
        doc_types=ALLOWED_DOC_TYPES,
        departments=ALLOWED_DEPARTMENTS,
        content_domains=ALLOWED_CONTENT_DOMAINS,
        routes=ALLOWED_ROUTES,
        excerpt=excerpt[:3000],
    )

    try:
        raw = str(get_llm("metadata_extraction").complete(prompt)).strip()

        try:
            extracted = json.loads(raw)
        except json.JSONDecodeError:
            cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            extracted = json.loads(cleaned)
    except Exception as exc:
        logger.warning("content metadata extraction failed, using defaults: %s", exc)
        return defaults

    result = dict(defaults)

    result["topic"] = str(extracted.get("topic") or defaults["topic"])[:60]
    result["summary"] = str(extracted.get("summary") or "")[:300]

    keywords = extracted.get("keywords")

    if isinstance(keywords, list):
        result["keywords"] = [str(keyword)[:40] for keyword in keywords[:5]]

    for field, allowed in (
        ("owning_department", ALLOWED_DEPARTMENTS),
        ("content_domain", ALLOWED_CONTENT_DOMAINS),
        ("intended_route", ALLOWED_ROUTES),
    ):
        value = extracted.get(field)

        if value in allowed:
            result[field] = value

    return result


def build_document_metadata(
    file_path: str,
    original_filename: str,
    parsed_with: str,
    frontmatter: dict,
    body: str,
    embed_model: str,
) -> dict:
    metadata = build_structural_metadata(file_path, original_filename, parsed_with, frontmatter)
    metadata.update(extract_content_metadata(body))
    metadata["embed_model"] = embed_model

    if frontmatter.get("department") in ALLOWED_DEPARTMENTS:
        metadata["owning_department"] = frontmatter["department"]

    return metadata
