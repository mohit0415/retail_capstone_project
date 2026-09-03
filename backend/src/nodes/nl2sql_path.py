import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from configs.llms import model_for
from configs.settings import settings
from src.auth.rbac import allowed_risk_categories, allowed_tables
from src.graph.state import AgentState
from src.observability.tracing import runnable_config, traced_node
from src.prompts.library import SQL_NARRATION
from src.schemas.models import DraftAnswer
from src.sqlpath.executor import SqlPolicyError, run_nl2sql, sanity_check

logger = logging.getLogger(__name__)


def _entity_hints(state: AgentState) -> str:
    hints = []

    for entity in state.get("resolved_entities", []):
        if entity.resolved_id is not None:
            hints.append(f"{entity.entity_type} \"{entity.surface_form}\" is id {entity.resolved_id}")
        elif entity.canonical_name:
            hints.append(f"{entity.entity_type} \"{entity.surface_form}\" is \"{entity.canonical_name}\"")

    if not hints:
        return ""

    return "\n\nResolved entities you must use rather than matching on the raw name:\n- " + "\n- ".join(hints)


def run_sql_evidence(state: AgentState):
    scopes = state.get("access_scopes", [])
    tables = allowed_tables(scopes)

    if not tables:
        return None, "this role has no access to the compliance database"

    question = state["standalone_query"] + _entity_hints(state)

    try:
        evidence = run_nl2sql(
            question=question,
            allowed_tables=tables,
            departments=state.get("departments", []),
            as_of=settings.as_of_date,
            risk_categories=allowed_risk_categories(scopes),
        )
    except SqlPolicyError as exc:
        logger.warning("nl2sql refused: %s", exc)
        return None, str(exc)
    except Exception as exc:
        logger.error("nl2sql failed: %s", exc)
        return None, "the database probe could not be completed"

    return evidence, ""


@traced_node("nl2sql_path")
def nl2sql_path_node(state: AgentState) -> dict:
    evidence, failure = run_sql_evidence(state)

    if evidence is None:
        return {
            "evidence_path": "nl2sql",
            "draft": DraftAnswer(
                answer=f"This question could not be answered from the compliance records: {failure}.",
                uncertainty_note="no database evidence was produced",
            ),
            "tokens_spent": 900,
        }

    caveats = sanity_check(evidence)

    prompt = SQL_NARRATION.format(
        query=state["standalone_query"],
        template_id=evidence.template_id,
        as_of=evidence.as_of,
        row_count=evidence.row_count,
        rows=json.dumps(evidence.rows[:25], default=str, indent=2),
        statement=evidence.statement,
        caveats="\n".join(f"- {item}" for item in caveats) or "- none",
    )

    model = model_for("nl2sql_intent").with_structured_output(DraftAnswer)

    draft: DraftAnswer = model.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=state["standalone_query"])],
        config=runnable_config(state, "sql_narration"),
    )

    return {
        "evidence_path": "nl2sql",
        "sql_evidence": evidence,
        "draft": draft,
        "tokens_spent": 1500,
    }
