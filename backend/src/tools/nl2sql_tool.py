import json
import logging
from typing import List

from llama_index.core.tools import FunctionTool

from configs.settings import settings
from src.sqlpath.executor import SqlPolicyError, run_nl2sql, sanity_check
from src.sqlpath.templates import tables_visible_to

logger = logging.getLogger(__name__)

SQL_TOOL_DESCRIPTION = (
    "Read-only natural-language query over the retail compliance database. It holds only these "
    "tables: vendors (supplier master with compliance_status, approval_status, contract dates, "
    "whether the vendor handles personal data), retention_records (records under a retention rule "
    "with retention_until and disposal_status), compliance_reviews (per-vendor review outcomes and "
    "open remediation), and audit_logs (the system's own append-only event log). "
    "Call this tool when the question asks about the state of a specific record: is this vendor "
    "approved, which contracts have expired, what did the last review find, are any records past "
    "their retention deadline, how many rows match a condition. "
    "Every figure is pinned to the as-of date, not the wall clock. "
    "Do not call this tool for what a policy says or requires; that lives in policy_documents. "
    "This tool cannot change anything, and a question that asks it to will be refused."
)


def build_sql_tool(
    allowed_tables: set[str],
    departments: List[str] | None = None,
    risk_categories: set[str] | None = None,
):
    departments = departments or []
    visible = tables_visible_to(allowed_tables)

    def query_compliance_records(input: str) -> str:
        if not visible:
            return "This role has no access to the compliance database."

        try:
            evidence = run_nl2sql(
                question=input,
                allowed_tables=allowed_tables,
                departments=departments,
                as_of=settings.as_of_date,
                risk_categories=risk_categories,
            )
        except SqlPolicyError as exc:
            logger.warning("nl2sql tool refused a call: %s", exc)
            return f"The database query was refused: {exc}"
        except Exception as exc:
            logger.error("nl2sql tool failed: %s", exc)
            return f"The database query could not be completed: {exc}"

        caveats = sanity_check(evidence)

        payload = {
            "sql": evidence.statement,
            "as_of": str(evidence.as_of),
            "row_count": evidence.row_count,
            "rows": evidence.rows[:25],
            "caveats": caveats,
        }

        return json.dumps(payload, default=str, indent=2)

    return FunctionTool.from_defaults(
        fn=query_compliance_records,
        name="compliance_records",
        description=SQL_TOOL_DESCRIPTION,
    )
