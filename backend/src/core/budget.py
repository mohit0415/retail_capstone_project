import logging
import time
from dataclasses import dataclass

from configs.settings import settings

logger = logging.getLogger(__name__)

OPTIONAL_TIER_NODES = {"reranker", "confidence_calibration", "thread_summary"}

PATH_DEADLINE_FIELD = {
    "rag": "deadline_seconds_standard",
    "nl2sql": "deadline_seconds_standard",
    "hybrid": "deadline_seconds_hybrid",
    "agentic": "deadline_seconds_agentic",
    "high_risk_panel": "deadline_seconds_high_risk",
}

# confidence_scoring is arithmetic over what validation already produced (no model call),
# so stopping it only turned a finished, validated answer into an escalation
RELEASE_TIER_NODES = {
    "output_guardrail",
    "escalation_manager",
    "safe_refusal",
    "clarification",
    "confidence_scoring",
    "no_answer",
}

GRACE_ELIGIBLE_NODES = {"compliance_validation"}

NODE_TOKEN_ESTIMATE = {
    "query_rewrite": 600,
    "intent_classification": 500,
    "entity_resolution": 300,
    "risk_assessment": 700,
    "planner": 800,
    "rag_path": 3000,
    "nl2sql_path": 1200,
    "hybrid_path": 4000,
    "agentic_rag": 5000,
    "multi_agent_panel": 7000,
    "compliance_validation": 2500,
    "reflection": 900,
}


@dataclass(slots=True)
class BudgetVerdict:
    allowed: bool
    reason: str = ""
    degrade: bool = False


class BudgetGuard:
    def __init__(self, deadline_ts: float, token_budget: int, tokens_spent: int):
        self.deadline_ts = deadline_ts
        self.token_budget = token_budget
        self.tokens_spent = tokens_spent

    @property
    def seconds_remaining(self) -> float:
        return self.deadline_ts - time.monotonic()

    @property
    def tokens_remaining(self) -> int:
        return self.token_budget - self.tokens_spent

    def check(self, node_name: str, grace_seconds: float = 0.0) -> BudgetVerdict:
        if node_name in RELEASE_TIER_NODES:
            return BudgetVerdict(allowed=True)

        estimate = NODE_TOKEN_ESTIMATE.get(node_name, 500)
        optional = node_name in OPTIONAL_TIER_NODES

        if self.seconds_remaining <= -max(0.0, grace_seconds):
            if optional:
                return BudgetVerdict(allowed=False, reason="deadline exhausted", degrade=True)

            return BudgetVerdict(allowed=False, reason="deadline exhausted")

        if self.tokens_remaining < estimate:
            if optional:
                return BudgetVerdict(allowed=False, reason="token budget exhausted", degrade=True)

            return BudgetVerdict(
                allowed=False,
                reason=f"token budget exhausted, {self.tokens_remaining} left and this step needs {estimate}",
            )

        if optional and self.seconds_remaining < 1.0:
            return BudgetVerdict(allowed=False, reason="insufficient headroom for optional node", degrade=True)

        return BudgetVerdict(allowed=True)


def finishing_grace_seconds(node_name: str, state: dict) -> float:
    """Extra seconds a node may start after the deadline.

    Only the validator of an answer that is already drafted gets it: escalating a
    finished draft to a human because validation would start a few seconds late is
    a far worse outcome than a response that is a few seconds slower. A High-risk
    answer goes to a human anyway, so it gets no grace.
    """
    if node_name not in GRACE_ELIGIBLE_NODES or state.get("draft") is None:
        return 0.0

    risk = state.get("risk")
    level = getattr(getattr(risk, "final_level", None), "value", None)

    if level == "High":
        return 0.0

    return max(0.0, float(settings.validation_grace_seconds))


def guard_from_state(state: dict) -> BudgetGuard:
    return BudgetGuard(
        deadline_ts=state.get("deadline_ts", time.monotonic() + settings.deadline_seconds_standard),
        token_budget=state.get("token_budget", settings.default_token_budget),
        tokens_spent=state.get("tokens_spent", 0),
    )


def path_deadline_seconds(path: str) -> float:
    field = PATH_DEADLINE_FIELD.get(path, "deadline_seconds_standard")

    return float(getattr(settings, field))


def widened_deadline(state: dict, seconds: float) -> float | None:
    started = state.get("started_ts")

    if started is None:
        return None

    candidate = started + seconds

    if candidate <= state.get("deadline_ts", candidate):
        return None

    return candidate


def widen_for_path(state: dict, path: str) -> dict:
    seconds = path_deadline_seconds(path)
    widened = widened_deadline(state, seconds)

    if widened is None:
        logger.debug("deadline unchanged for path=%s (already >= %.0fs)", path, seconds)

        return {}

    logger.info(
        "deadline widened path=%s to %.0fs from start (was %.1fs) request_id=%s",
        path,
        seconds,
        state.get("deadline_ts", 0.0) - state.get("started_ts", 0.0),
        state.get("request_id"),
    )

    return {"deadline_ts": widened}
