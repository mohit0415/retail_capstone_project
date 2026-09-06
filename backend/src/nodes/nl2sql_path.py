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
from src.sqlpath.executor import SqlPolicyError, run_vetted_sql, sanity_check

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

    return "\n- ".join(["", *hints]).strip()


def _presets(state: AgentState) -> dict:
    for entity in state.get("resolved_entities", []):
        if entity.entity_type == "vendor" and entity.resolved_id is not None:
            return {"vendor_id": entity.resolved_id}

    return {}


def run_sql_evidence(state: AgentState):
    scopes = state.get("access_scopes", [])
    tables = allowed_tables(scopes)

    if not tables:
        return None, "this role has no access to the compliance database"

    try:
        evidence = run_vetted_sql(
            question=state["standalone_query"],
            allowed_tables=tables,
            departments=state.get("departments", []),
            as_of=settings.as_of_date,
            risk_categories=allowed_risk_categories(scopes),
            presets=_presets(state),
            entity_hints=_entity_hints(state),
            config=runnable_config(state, "sql_template_selector"),
        )
    except SqlPolicyError as exc:
        logger.warning("vetted sql refused: %s", exc)
        return None, str(exc)
    except Exception as exc:
        logger.error("vetted sql failed: %s", exc)
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
            "escalation_reason": f"the records path produced no evidence: {failure}",
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
