from langchain_core.messages import HumanMessage, SystemMessage

from configs.llms import model_for
from configs.settings import settings
from src.auth.rbac import allowed_doc_types, allowed_tables
from src.graph.state import AgentState
from src.observability.tracing import runnable_config, traced_node
from src.prompts.library import PLANNER
from src.schemas.enums import EntityStatus, EvidencePath, Intent, RiskLevel
from src.schemas.models import EvidencePlan, PlanStep

INTENT_DEFAULT_PATH = {
    Intent.POLICY_LOOKUP: EvidencePath.RAG,
    Intent.RECORD_LOOKUP: EvidencePath.NL2SQL,
    Intent.VENDOR_STATUS: EvidencePath.NL2SQL,
    Intent.RETENTION_QUERY: EvidencePath.HYBRID,
    Intent.COMPLIANCE_CHECK: EvidencePath.HYBRID,
    Intent.INCIDENT_GUIDANCE: EvidencePath.AGENTIC,
}


def _fallback_plan(state: AgentState, path: EvidencePath) -> EvidencePlan:
    return EvidencePlan(
        path=path,
        steps=[
            PlanStep(
                order=1,
                source="both" if path is not EvidencePath.RAG else "policy_kb",
                objective="gather the evidence the question turns on",
                must_prove="the answer is supported by a retrieved clause or a returned row",
            )
        ],
        required_claims=[],
        revision=state.get("plan").revision + 1 if state.get("plan") else 0,
    )


@traced_node("planner")
def planner_node(state: AgentState) -> dict:
    risk = state["risk"]
    intent = state.get("intent")

    if risk.final_level is RiskLevel.HIGH:
        forced_path = EvidencePath.HIGH_RISK_PANEL
    else:
        forced_path = INTENT_DEFAULT_PATH.get(intent.intent if intent else None, EvidencePath.RAG)

    scopes = state.get("access_scopes", [])
    docs = allowed_doc_types(scopes)
    tables = sorted(allowed_tables(scopes))

    unresolved = [
        entity.surface_form
        for entity in state.get("resolved_entities", [])
        if entity.status is not EntityStatus.EXACT
    ]

    previous = state.get("plan")
    revision_context = ""

    if previous is not None and state.get("validation") is not None:
        defects = "; ".join(d.description for d in state["validation"].defects)
        revision_context = (
            f"This is a re-plan. The previous plan produced these defects: {defects}. "
            "The new plan must gather evidence the previous one missed."
        )

    prompt = PLANNER.format(
        risk_level=risk.final_level.value,
        allowed_docs=", ".join(docs) or "none",
        allowed_tables=", ".join(tables) or "none",
        unresolved=", ".join(unresolved) or "none",
        revision_context=revision_context,
        query=state["standalone_query"],
    )

    model = model_for("planner").with_structured_output(EvidencePlan)

    try:
        plan: EvidencePlan = model.invoke(
            [SystemMessage(content=prompt), HumanMessage(content=state["standalone_query"])],
            config=runnable_config(state, "planner"),
        )
    except Exception:
        plan = _fallback_plan(state, forced_path)

    if risk.final_level is RiskLevel.HIGH:
        plan.path = EvidencePath.HIGH_RISK_PANEL

    if not tables and plan.path in (EvidencePath.NL2SQL, EvidencePath.HYBRID):
        plan.path = EvidencePath.RAG

    if plan.path is EvidencePath.AGENTIC and not settings.enable_agentic_path:
        plan.path = EvidencePath.HYBRID if tables else EvidencePath.RAG

    plan.revision = previous.revision + 1 if previous else 0

    return {"plan": plan, "tokens_spent": 800}
