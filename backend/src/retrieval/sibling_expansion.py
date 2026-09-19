"""Sibling-clause expansion: numbered sections reach the writer complete.

The cross-encoder judges every clause chunk in isolation, so a clause whose
meaning lives mostly in its heading is routinely cut: "6.2 Cross-Border
Transfers" bodies never say "data sharing", and a question about data sharing
restrictions then gets an answer built from §6.1 alone. After reranking, the
top clauses pull in the missing siblings of their numbered section
(6.1 -> every current 6.x of the same document) so the writer sees the whole
section, not the half the reranker happened to keep.

Scope safety: siblings come from the same document as the seed chunk and the
query is still filtered by the caller's allowed doc types, so this can never
surface a document the role could not already retrieve. Expansion failures
only log - retrieval must never break because of it.
"""

import logging
import re

from configs.database import read_only_connection
from configs.settings import settings
from src.retrieval.adapter import _parse_date
from src.schemas.models import RetrievedChunk

logger = logging.getLogger(__name__)

# a sibling clause is a short rule; anything longer is truncated in the appended sentence
SIBLING_BODY_MAX_CHARS = 400

# the llama-index PGVectorStore stores rows under "data_" + configured name
SIBLINGS_SQL_TEMPLATE = """
SELECT node_id, text, metadata_
FROM data_{table}
WHERE metadata_->>'document_title' = %(title)s
  AND metadata_->>'version' = %(version)s
  AND metadata_->>'doc_type' = ANY(%(doc_types)s)
  AND metadata_->>'content_type' = 'text'
  AND metadata_->>'clause_number' LIKE %(pattern)s
ORDER BY metadata_->>'clause_number', metadata_->>'chunk_index'
LIMIT 20
"""


def _parent_clause(clause_number: str) -> str | None:
    """"6.1" -> "6"; a top-level clause ("6", "Fig-p3") has no numbered parent."""
    head, dot, tail = (clause_number or "").rpartition(".")

    return head if dot and head and tail else None


def _to_chunk(row: dict) -> RetrievedChunk:
    metadata = row["metadata_"] or {}

    return RetrievedChunk(
        chunk_id=row["node_id"],
        doc_type=metadata.get("doc_type", "unknown"),
        document_title=metadata.get("document_title", "Untitled"),
        section=metadata.get("section", "general"),
        clause_number=str(metadata.get("clause_number", "0")),
        version=metadata.get("version", "1.0"),
        effective_date=_parse_date(metadata.get("effective_date")),
        content=row["text"],
        content_type=metadata.get("content_type", "text"),
        modality=metadata.get("modality", "text"),
    )


def expand_sibling_clauses(
    chunks: list[RetrievedChunk], allowed_doc_types: list[str]
) -> list[RetrievedChunk]:
    """Append the missing sibling clauses of the top-ranked chunks' sections."""
    if not chunks or not allowed_doc_types:
        return chunks

    seeds = [c for c in chunks[: settings.sibling_expansion_seeds] if c.content_type == "text"]
    have = {(c.document_title, c.clause_number) for c in chunks}
    added: list[RetrievedChunk] = []
    sql = SIBLINGS_SQL_TEMPLATE.format(table=settings.vector_table_name)

    for seed in seeds:
        parent = _parent_clause(seed.clause_number)

        if parent is None:
            continue

        try:
            with read_only_connection() as conn:
                rows = conn.execute(
                    sql,
                    {
                        "title": seed.document_title,
                        "version": seed.version,
                        "doc_types": list(allowed_doc_types),
                        "pattern": f"{parent}.%",
                    },
                ).fetchall()
        except Exception as exc:
            logger.warning("sibling-clause expansion failed for %s: %s", seed.citation, exc)

            continue

        for row in rows:
            if len(added) >= settings.max_sibling_clauses:
                break

            chunk = _to_chunk(dict(row))
            key = (chunk.document_title, chunk.clause_number)

            if key in have:
                continue

            have.add(key)
            added.append(chunk)

    if added:
        logger.info(
            "sibling-clause expansion added %d clause(s): %s",
            len(added),
            ", ".join(c.citation for c in added),
        )

    return chunks + added


def append_sibling_clauses(
    answer: str, cited_clauses: list[str], chunks: list[RetrievedChunk]
) -> tuple[str, list[str]]:
    """Append the supplied sibling clauses the writer left out of the answer.

    The small writer model reliably answers with the single closest clause even
    when told not to (citing 6.1 with 6.2 supplied), so section completeness is
    enforced here deterministically: for every cited clause, each supplied
    extract from the same numbered section that the answer does not mention is
    appended verbatim with its citation. Returns the new answer and the added
    citations (empty when nothing was missing).
    """
    if not answer or not cited_clauses or not chunks:
        return answer, []

    by_citation = {c.citation: c for c in chunks}
    added: list[str] = []

    for cited in cited_clauses:
        seed = by_citation.get(cited.strip("[]"))

        if seed is None:
            continue

        parent = _parent_clause(seed.clause_number)

        if parent is None:
            continue

        for chunk in chunks:
            if len(added) >= settings.max_sibling_clauses:
                break

            if (
                chunk.document_title != seed.document_title
                or chunk.clause_number == seed.clause_number
                or _parent_clause(chunk.clause_number) != parent
                or chunk.citation in answer
                or chunk.citation in added
            ):
                continue

            body = re.sub(r"\s+", " ", chunk.content).strip()

            if len(body) > SIBLING_BODY_MAX_CHARS:
                body = body[: SIBLING_BODY_MAX_CHARS - 1].rstrip() + "…"

            answer = f"{answer}\n\nIn the same section, {chunk.section}: {body} [{chunk.citation}]"
            added.append(chunk.citation)

    return answer, added
