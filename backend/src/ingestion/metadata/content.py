import json
import logging

from src.index.models import get_llm
from src.ingestion.metadata.schema import (
    ALLOWED_CONTENT_DOMAINS,
    ALLOWED_DEPARTMENTS,
    ALLOWED_DOC_TYPES,
    ALLOWED_ROUTES,
)

logger = logging.getLogger(__name__)

EXCERPT_CHARS = 3000

CONTENT_PROMPT = """You are tagging a document from a retail compliance corpus made of five internal
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

DEFAULTS = {
    "topic": "unknown",
    "keywords": [],
    "owning_department": "all",
    "content_domain": "mixed",
    "intended_route": "rag",
    "summary": "",
}


def _decode(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        return json.loads(cleaned)


def extract_content_metadata(body: str) -> dict:
    prompt = CONTENT_PROMPT.format(
        doc_types=ALLOWED_DOC_TYPES,
        departments=ALLOWED_DEPARTMENTS,
        content_domains=ALLOWED_CONTENT_DOMAINS,
        routes=ALLOWED_ROUTES,
        excerpt=(body or "")[:EXCERPT_CHARS],
    )

    try:
        extracted = _decode(str(get_llm("metadata_extraction").complete(prompt)).strip())
    except Exception as exc:
        logger.warning("content metadata extraction failed, falling back to defaults: %s", exc)

        return dict(DEFAULTS)

    result = dict(DEFAULTS)

    result["topic"] = str(extracted.get("topic") or DEFAULTS["topic"])[:60]
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

    inferred_doc_type = extracted.get("doc_type")

    if inferred_doc_type in ALLOWED_DOC_TYPES:
        result["inferred_doc_type"] = inferred_doc_type

    return result
