import logging
import re
from datetime import date

from llama_index.core.schema import NodeWithScore

from src.schemas.models import RetrievedChunk

logger = logging.getLogger(__name__)

# a figure's caption node carries element_label "page3_fig0" and no clause number
FIGURE_LABEL = re.compile(r"page(\d+)_fig(\d+)")


def _clause_number(metadata: dict) -> str:
    """The clause an extract is cited by; a figure is cited by the page it sits on.

    A figure (a flowchart's caption) has no clause of its own, so it used to be cited by the document
    title alone - "[Anti Bribery Ethical Conduct Policy]", which is not a [Document §clause] marker.
    A claim only the flowchart supports could then never be cited, failed validation and escalated.
    It is now "§Fig-p3" (the first figure on page 3), citable like any clause.
    """
    clause = metadata.get("clause_number", "0")

    if clause:
        return str(clause)

    match = FIGURE_LABEL.search(str(metadata.get("element_label") or ""))

    if match:
        page, figure = int(match.group(1)), int(match.group(2))

        return f"Fig-p{page}" if figure == 0 else f"Fig-p{page}-{figure + 1}"

    if metadata.get("content_type") == "image_caption":
        return "Fig"

    return clause


def _parse_date(value) -> date | None:
    if isinstance(value, date):
        return value

    if not value:
        return None

    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        logger.debug("could not parse date %r, treating as no date", value)

        return None


def to_retrieved_chunk(scored: NodeWithScore, fused: bool, reranked: bool = False) -> RetrievedChunk:
    node = getattr(scored, "node", scored)
    metadata = getattr(node, "metadata", {}) or {}

    try:
        content = node.get_content()
    except Exception:
        logger.debug("node.get_content() failed, falling back to str(node)", exc_info=True)
        content = str(node)

    score = float(scored.score) if getattr(scored, "score", None) is not None else 0.0

    chunk = RetrievedChunk(
        chunk_id=getattr(node, "node_id", "") or getattr(node, "id_", ""),
        doc_type=metadata.get("doc_type", "unknown"),
        document_title=metadata.get("document_title", "Untitled"),
        section=metadata.get("section", "general"),
        clause_number=_clause_number(metadata),
        version=metadata.get("version", "1.0"),
        effective_date=_parse_date(metadata.get("effective_date")),
        content=content,
        content_type=metadata.get("content_type", "text"),
        modality=metadata.get("modality", "text"),
        image_path=metadata.get("image_path"),
        original_table=metadata.get("original_table"),
    )

    if reranked:
        chunk.rerank_score = score
    elif fused:
        chunk.fused_score = score
    else:
        chunk.dense_score = score
        chunk.fused_score = score

    return chunk


def to_retrieved_chunks(
    nodes: list[NodeWithScore],
    fused: bool,
    reranked: bool = False,
) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    seen: set[str] = set()

    for scored in nodes:
        chunk = to_retrieved_chunk(scored, fused, reranked)

        if chunk.chunk_id in seen:
            continue

        seen.add(chunk.chunk_id)
        chunks.append(chunk)

    return chunks


def provenance_text(chunk: RetrievedChunk) -> str:
    """One chunk as the RAGAS judge should see it: provenance first, then body.

    A judge given bare text counts the answer's citation markers ("[Title §N]")
    as unsupported claims, deflating faithfulness on every request.
    """
    if chunk.section:
        return f"{chunk.citation} ({chunk.section}): {chunk.content}"

    return f"{chunk.citation}: {chunk.content}"


def format_context(chunks: list[RetrievedChunk]) -> str:
    from src.guardrails.injection import neutralise_retrieved
    from src.retrieval.citations import sub_clause_headings

    blocks = []

    for chunk in chunks:
        header = (
            f"[{chunk.citation}] "
            f"section: {chunk.section} | version: {chunk.version} | modality: {chunk.modality}"
        )

        inner = [number for number, _heading, _offset in sub_clause_headings(chunk)]

        if inner:
            # the numbered headings inside this extract can be cited as [<title> §<number>]
            header += " | clauses inside: " + ", ".join(f"§{number}" for number in inner)

        body = chunk.content

        if chunk.original_table:
            body = f"{body}\n\nOriginal table:\n{chunk.original_table}"

        blocks.append(neutralise_retrieved(f"{header}\n{body}", source=chunk.citation))

    return "\n\n".join(blocks) if blocks else "(no policy extract was retrieved)"
