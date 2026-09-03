import logging
from typing import List

from llama_index.core.schema import TextNode

from configs.settings import settings
from src.index.models import get_llm
from src.ingestion.elements import RawTable
from src.ingestion.metadata.stamping import apply_exclusions, table_node_metadata

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """You are summarising a table taken from a retail policy or compliance document.

State what the table establishes. Keep every number, every date, every retention period and
every threshold exactly as printed. Name the columns. If the table sets a rule or a limit,
say what the rule is.

Do not interpret the table's consequences and do not add anything that is not in it.

Table:
{table}

Summary:"""


def structural_summary(table_markdown: str) -> str:
    lines = [line for line in table_markdown.splitlines() if line.strip()]
    data_rows = max(0, len(lines) - 2)

    header = ""

    for line in lines[:2]:
        if "|" in line and not all(character in "|-: " for character in line.strip()):
            header = line.strip()
            break

    summary = f"Table with {data_rows} data row(s)"

    if header:
        summary += f" and columns: {header[:250]}"

    return summary + ". The full table is preserved in node metadata."


class TableProcessor:
    def __init__(self):
        self.llm = None

    def _summarise(self, table_markdown: str) -> str:
        truncated = table_markdown[: settings.max_table_chars]

        if len(table_markdown) > settings.max_table_chars:
            truncated += "\n[table truncated for summarisation]"

        if self.llm is None:
            self.llm = get_llm("table_summary")

        try:
            return str(self.llm.complete(SUMMARY_PROMPT.format(table=truncated))).strip()
        except Exception as exc:
            logger.warning("table summarisation failed, using the structural fallback: %s", exc)

            return structural_summary(truncated)

    def process(self, tables: List[RawTable], document_metadata: dict) -> List[TextNode]:
        nodes: List[TextNode] = []

        for table in tables:
            summary = self._summarise(table.markdown)

            if not summary:
                summary = structural_summary(table.markdown)

            metadata = table_node_metadata(document_metadata, table)

            nodes.append(apply_exclusions(TextNode(text=summary, metadata=metadata)))

        logger.info("built %s table node(s)", len(nodes))

        return nodes
