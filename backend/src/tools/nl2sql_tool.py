import json
import logging

from llama_index.core.tools import FunctionTool

from configs.settings import settings
from src.sqlpath.executor import SqlPolicyError, run_vetted_sql, sanity_check
from src.sqlpath.templates import catalogue_for, templates_visible_to


def _tool_description(catalogue: str) -> str:
    return (
        "Read-only lookup over the retail compliance database. It does not accept SQL and it does "
        "not write SQL. It matches your question to one of a fixed set of reviewed queries and runs "
        "that query with bound parameters, so anything outside the list below cannot be asked. "
        "Every figure is pinned to the as-of date, not the wall clock. "
        "Use it for the state of a record: is this vendor approved, which findings are overdue, "
        "what did the last review say, which retention records need review. "
        "Do not use it for what a policy says or requires; that lives in policy_documents.\n\n"
        "Reviewed queries available to you:\n"
        f"{catalogue}"
    )


logger = logging.getLogger(__name__)


def build_sql_tool(
    allowed_tables: set[str],
    departments: list[str] | None = None,
    risk_categories: set[str] | None = None,
):
    departments = departments or []
    catalogue = templates_visible_to(allowed_tables)

    def query_compliance_records(input: str) -> str:
        if not catalogue:
            return "This role has no access to the compliance database."

        try:
            evidence = run_vetted_sql(
                question=input,
                allowed_tables=allowed_tables,
                departments=departments,
                as_of=settings.as_of_date,
                risk_categories=risk_categories,
            )
        except SqlPolicyError as exc:
            logger.warning("vetted sql tool refused a call: %s", exc)
            return f"The database query was refused: {exc}"
        except Exception as exc:
            logger.error("vetted sql tool failed: %s", exc)
            return f"The database query could not be completed: {exc}"

        caveats = sanity_check(evidence)

        payload = {
            "template_id": evidence.template_id,
            "sql": evidence.statement,
            "parameters": evidence.parameters,
            "as_of": str(evidence.as_of),
            "row_count": evidence.row_count,
            "rows": evidence.rows[:25],
            "caveats": caveats,
        }

        return json.dumps(payload, default=str, indent=2)

    return FunctionTool.from_defaults(
        fn=query_compliance_records,
        name="compliance_records",
        description=_tool_description(catalogue_for(catalogue)),
    )
