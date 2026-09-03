from configs.settings import settings
from src.core.budget import guard_from_state
from src.graph.state import AgentState
from src.schemas.enums import EvidencePath, RiskLevel, TerminalOutcome


def after_guardrail(state: AgentState) -> str:
    if state.get("terminal_outcome") == TerminalOutcome.REFUSED.value:
        return "refuse"

    return "continue"


def after_intent(state: AgentState) -> str:
    if state.get("terminal_outcome") == TerminalOutcome.REFUSED.value:
        return "refuse"

    return "continue"


def after_entity_resolution(state: AgentState) -> str:
    if state.get("terminal_outcome") == TerminalOutcome.CLARIFICATION_REQUIRED.value:
        return "clarify"

    return "continue"


def after_risk(state: AgentState) -> str:
    if state.get("escalation_reason"):
        return "escalate"

    return "continue"


def route_evidence_path(state: AgentState) -> str:
    plan = state.get("plan")
    risk = state.get("risk")

    if risk and risk.final_level is RiskLevel.HIGH:
        return EvidencePath.HIGH_RISK_PANEL.value

    if plan is None:
        return EvidencePath.RAG.value

    guard = guard_from_state(state)

    if plan.path is EvidencePath.AGENTIC and guard.seconds_remaining < settings.agentic_min_seconds:
        return EvidencePath.HYBRID.value

    if plan.path is EvidencePath.HYBRID and guard.seconds_remaining < 1.5:
        return EvidencePath.RAG.value

    return plan.path.value


def after_validation(state: AgentState) -> str:
    validation = state.get("validation")

    if validation is None:
        return "score"

    if validation.passed:
        return "score"

    if state.get("reflection_count", 0) >= settings.max_reflection_retries:
        return "escalate"

    guard = guard_from_state(state)

    if guard.seconds_remaining < 1.0 or guard.tokens_remaining < 2000:
        return "escalate"

    return "reflect"


def after_confidence(state: AgentState) -> str:
    confidence = state.get("confidence")
    risk = state.get("risk")
    validation = state.get("validation")
    panel = state.get("panel_verdict")

    if risk and risk.final_level is RiskLevel.HIGH:
        return "escalate"

    if validation and validation.conflict_detected:
        return "escalate"

    if panel and panel.unresolved_conflict:
        return "escalate"

    if confidence is None or confidence.final_score < settings.confidence_threshold:
        return "escalate"

    return "respond"
