from configs.settings import settings
from src.core.budget import guard_from_state
from src.graph.path_policy import ESCALATE, degrade_for_budget
from src.graph.state import AgentState
from src.schemas.enums import EvidencePath, ReviewDecision, RiskLevel, TerminalOutcome

REPAIR_HEADROOM_SECONDS = 1.0

REPAIR_HEADROOM_TOKENS = 2000

RESUMABLE_DECISIONS = {ReviewDecision.ACCEPT.value, ReviewDecision.EDIT.value}


def _stopped(state: AgentState) -> bool:
    return bool(state.get("escalation_reason")) or bool(state.get("budget_stops"))


def after_guardrail(state: AgentState) -> str:
    if state.get("terminal_outcome") == TerminalOutcome.REFUSED.value:
        return "refuse"

    if _stopped(state):
        return "escalate"

    return "continue"


def after_intent(state: AgentState) -> str:
    if state.get("terminal_outcome") == TerminalOutcome.REFUSED.value:
        return "refuse"

    if _stopped(state):
        return "escalate"

    return "continue"


def after_entity_resolution(state: AgentState) -> str:
    if state.get("terminal_outcome") == TerminalOutcome.CLARIFICATION_REQUIRED.value:
        return "clarify"

    if _stopped(state):
        return "escalate"

    return "continue"


def after_risk(state: AgentState) -> str:
    if _stopped(state):
        return "escalate"

    return "continue"


def route_evidence_path(state: AgentState) -> str:
    if _stopped(state):
        return "escalate"

    risk = state.get("risk")

    if risk and risk.final_level is RiskLevel.HIGH:
        return EvidencePath.HIGH_RISK_PANEL.value

    routed = state.get("routed_path")

    if routed == ESCALATE:
        return "escalate"

    if routed is None:
        plan = state.get("plan")
        routed = plan.path.value if plan else EvidencePath.RAG.value

    guard = guard_from_state(state)
    intent = state["intent"].intent if state.get("intent") else None

    decision = degrade_for_budget(
        path=EvidencePath(routed),
        intent=intent,
        seconds_remaining=guard.seconds_remaining,
        agentic_min_seconds=settings.agentic_min_seconds,
        hybrid_min_seconds=settings.hybrid_min_seconds,
    )

    return decision.path


def after_evidence(state: AgentState) -> str:
    if _stopped(state):
        return "escalate"

    return "continue"


def after_validation(state: AgentState) -> str:
    if state.get("budget_stops"):
        return "escalate"

    validation = state.get("validation")

    if validation is None:
        return "score"

    if validation.passed:
        return "score"

    if state.get("reflection_count", 0) >= settings.max_reflection_retries:
        return "escalate"

    guard = guard_from_state(state)

    if guard.seconds_remaining < REPAIR_HEADROOM_SECONDS:
        return "escalate"

    if guard.tokens_remaining < REPAIR_HEADROOM_TOKENS:
        return "escalate"

    return "reflect"


def after_confidence(state: AgentState) -> str:
    confidence = state.get("confidence")
    risk = state.get("risk")
    validation = state.get("validation")
    panel = state.get("panel_verdict")

    if _stopped(state):
        return "escalate"

    if risk and risk.final_level is RiskLevel.HIGH:
        return "escalate"

    if validation and validation.conflict_detected:
        return "escalate"

    if panel and panel.unresolved_conflict:
        return "escalate"

    if confidence is None or confidence.final_score < settings.confidence_threshold:
        return "escalate"

    return "respond"


def after_output_guardrail(state: AgentState) -> str:
    if state.get("terminal_outcome") != TerminalOutcome.ESCALATED.value:
        return "done"

    if state.get("reviewer_decision"):
        return "done"

    return "escalate"


def after_escalation(state: AgentState) -> str:
    if state.get("reviewer_decision") in RESUMABLE_DECISIONS:
        return "review"

    return "wait"
