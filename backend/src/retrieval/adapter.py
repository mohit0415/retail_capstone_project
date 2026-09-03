from datetime import date
from typing import List

from llama_index.core.schema import NodeWithScore

from src.schemas.models import RetrievedChunk


def _parse_date(value) -> date | None:
    if isinstance(value, date):
        return value

    if not value:
        return None

    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def to_retrieved_chunk(scored: NodeWithScore, fused: bool) -> RetrievedChunk:
    node = getattr(scored, "node", scored)
    metadata = getattr(node, "metadata", {}) or {}

    try:
        content = node.get_content()
    except Exception:
        content = str(node)

    score = float(scored.score) if getattr(scored, "score", None) is not None else 0.0

    chunk = RetrievedChunk(
        chunk_id=getattr(node, "node_id", "") or getattr(node, "id_", ""),
        doc_type=metadata.get("doc_type", "unknown"),
        document_title=metadata.get("document_title", "Untitled"),
        section=metadata.get("section", "general"),
        clause_number=metadata.get("clause_number", "0"),
        version=metadata.get("version", "1.0"),
        effective_date=_parse_date(metadata.get("effective_date")),
        content=content,
        content_type=metadata.get("content_type", "text"),
        modality=metadata.get("modality", "text"),
        image_path=metadata.get("image_path"),
        original_table=metadata.get("original_table"),
    )

    if fused:
        chunk.fused_score = score
    else:
        chunk.dense_score = score
        chunk.fused_score = score

    return chunk


def to_retrieved_chunks(nodes: List[NodeWithScore], fused: bool) -> List[RetrievedChunk]:
    chunks: List[RetrievedChunk] = []
    seen: set[str] = set()

    for scored in nodes:
        chunk = to_retrieved_chunk(scored, fused)

        if chunk.chunk_id in seen:
            continue

        seen.add(chunk.chunk_id)
        chunks.append(chunk)

    return chunks


def format_context(chunks: List[RetrievedChunk]) -> str:
    from src.guardrails.injection import neutralise_retrieved

    blocks = []

    for chunk in chunks:
        header = (
            f"[{chunk.document_title} §{chunk.clause_number}] "
            f"section: {chunk.section} | version: {chunk.version} | modality: {chunk.modality}"
        )

        body = chunk.content

        if chunk.original_table:
            body = f"{body}\n\nOriginal table:\n{chunk.original_table}"

        blocks.append(neutralise_retrieved(f"{header}\n{body}", source=chunk.chunk_id))

    return "\n\n".join(blocks) if blocks else "(no policy extract was retrieved)"
