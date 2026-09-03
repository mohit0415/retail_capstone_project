import logging
import re
from typing import List

from llama_index.core.schema import TextNode

from configs.settings import settings
from src.index.models import get_llm

logger = logging.getLogger(__name__)

PIPE_TABLE = re.compile(
    r"^[ \t]{0,3}\|?[^\n|]*\|.*\r?\n"
    r"^[ \t]{0,3}(?=[^\n]*-{2,})(?=[^\n]*\|)[-:|\t ]+\r?\n"
    r"(?:^[ \t]{0,3}\|?[^\n|]*\|.*(?:\r?\n|$))+",
    re.M,
)

HTML_TABLE = re.compile(r"<table\b[^>]*>.*?</table>", re.S | re.I)

TABLE_SUMMARY_PROMPT = """You are summarising a table taken from a retail policy or compliance document.

State what the table establishes. Keep every number, every date, every retention period and
every threshold exactly as printed. Name the columns. If the table sets a rule or a limit,
say what the rule is.

Do not interpret the table's consequences and do not add anything that is not in it.

Table:
{table}

Summary:"""


def extract_markdown_tables(markdown: str) -> List[str]:
    found = [match.group(0).strip() for match in PIPE_TABLE.finditer(markdown)]

    found.extend(match.group(0).strip() for match in HTML_TABLE.finditer(markdown))

    if not found:
        logger.info("no tables found in the parsed markdown")

    return found


def _fallback_summary(table: str) -> str:
    lines = [line for line in table.splitlines() if line.strip()]
    row_count = max(0, len(lines) - 2)

    header = ""

    for line in lines[:2]:
        if "|" in line and not all(character in "|-: " for character in line.strip()):
            header = line.strip()
            break

    summary = f"Table with {row_count} data row(s)"

    if header:
        summary += f" and columns: {header[:250]}"

    return summary + ". The full table is preserved in node metadata."


def build_table_nodes(markdown: str, base_metadata: dict) -> List[TextNode]:
    tables = extract_markdown_tables(markdown)

    if not tables:
        return []

    llm = get_llm("table_summary")
    nodes: List[TextNode] = []

    for position, table in enumerate(tables):
        truncated = table[: settings.max_table_chars]

        if len(table) > settings.max_table_chars:
            truncated += "\n[table truncated for summarisation]"

        try:
            summary = str(llm.complete(TABLE_SUMMARY_PROMPT.format(table=truncated))).strip()
        except Exception as exc:
            logger.warning("table summarisation failed, using structural fallback: %s", exc)
            summary = _fallback_summary(truncated)

        metadata = {
            **base_metadata,
            "content_type": "table_summary",
            "modality": "table",
            "table_index": position,
            "element_label": f"table_{position}",
            "original_table": table,
        }

        nodes.append(TextNode(text=summary, metadata=metadata))

    return nodes
