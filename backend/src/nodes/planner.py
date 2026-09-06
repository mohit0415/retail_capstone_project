import logging

from langchain_core.messages import HumanMessage, SystemMessage

from configs.llms import model_for
from configs.settings import settings
from src.auth.rbac import allowed_doc_types, allowed_tables
from src.core.budget import widen_for_path
from src.graph.path_policy import ESCALATE, allowed_path_names, resolve_path, rule_for
from src.graph.state import AgentState
from src.observability.tracing import runnable_config, traced_node
from src.prompts.library import PLANNER
from src.schemas.enums import EntityStatus, EvidencePath
from src.schemas.models import EvidencePlan, PlanStep

logger = logging.getLogger(__name__)

SOURCE_FOR_EVIDENCE = {"policy": "policy_kb", "records": "compliance_db", "both": "both"}


def _fallback_plan(state: AgentState, path: EvidencePath) -> EvidencePlan:
    rule = rule_for(state["intent"].intent if state.get("intent") else None)

    return EvidencePlan(
        path=path,
        steps=[
            PlanStep(
                order=1,
                source=SOURCE_FOR_EVIDENCE[rule.evidence],
                objective="gather the evidence the question turns on",
                must_prove="the answer is supported by a retrieved clause or a returned row",
            )
        ],
        required_claims=[],
        revision=state.get("plan").revision + 1 if state.get("plan") else 0,
    )


def _unresolved_surfaces(state: AgentState) -> list[str]:
    return [
        entity.surface_form
        for entity in state.get("resolved_entities", [])
        if entity.status is not EntityStatus.EXACT
    ]


def _draft_plan(state: AgentState, intent, forced_path: EvidencePath) -> EvidencePlan:
    scopes = state.get("access_scopes", [])
    rule = rule_for(intent)

    prompt = PLANNER.format(
        risk_level=state["risk"].final_level.value,
        allowed_docs=", ".join(allowed_doc_types(scopes)) or "none",
        allowed_tables=", ".join(sorted(allowed_tables(scopes))) or "none",
        unresolved=", ".join(_unresolved_surfaces(state)) or "none",
        allowed_paths=", ".join(allowed_path_names(intent)),
        default_path=rule.default.value,
        path_rationale=rule.rationale,
        revision_context=state.get("replan_directive", ""),
        query=state["standalone_query"],
    )

    model = model_for("planner").with_structured_output(EvidencePlan)

    try:
        return model.invoke(
            [SystemMessage(content=prompt), HumanMessage(content=state["standalone_query"])],
            config=runnable_config(state, "planner"),
        )
    except Exception:
        return _fallback_plan(state, forced_path)


@traced_node("planner")
def planner_node(state: AgentState) -> dict:
    risk = state["risk"]
    intent = state["intent"].intent if state.get("intent") else None
    previous = state.get("plan")
    replanned = bool(state.get("plan_from_reflection"))

    if replanned and previous is not None:
        plan = previous
        tokens = 0
    else:
        plan = _draft_plan(state, intent, rule_for(intent).default)
        tokens = 800

    scopes = state.get("access_scopes", [])

    decision = resolve_path(
        intent=intent,
        risk_level=risk.final_level,
        proposed=plan.path,
        has_tables=bool(allowed_tables(scopes)),
        has_documents=bool(allowed_doc_types(scopes)),
        agentic_enabled=settings.enable_agentic_path,
    )

    logger.info(
        "route intent=%s proposed=%s chosen=%s clamped=%s replanned=%s reason=%s request_id=%s",
        intent.value if intent else "unknown",
        plan.path.value,
        decision.path,
        decision.clamped,
        replanned,
        decision.reason,
        state.get("request_id"),
    )

    if decision.path == ESCALATE:
        return {
            "plan": plan,
            "routed_path": ESCALATE,
            "path_decision": {"path": ESCALATE, "reason": decision.reason, "clamped": True},
            "escalation_reason": decision.reason,
            "plan_from_reflection": False,
            "tokens_spent": tokens,
        }

    plan.path = EvidencePath(decision.path)
    plan.revision = previous.revision + 1 if previous and not replanned else plan.revision

    result = {
        "plan": plan,
        "routed_path": decision.path,
        "path_decision": {
            "path": decision.path,
            "reason": decision.reason,
            "clamped": decision.clamped,
            "partial_evidence": decision.partial_evidence,
            "replanned": replanned,
        },
        "plan_from_reflection": False,
        "degraded": decision.partial_evidence or state.get("degraded", False),
        "tokens_spent": tokens,
    }
    result.update(widen_for_path(state, decision.path))

    return result
