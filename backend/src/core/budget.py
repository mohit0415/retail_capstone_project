import time
from dataclasses import dataclass

from configs.settings import settings

OPTIONAL_TIER_NODES = {"reranker", "confidence_calibration", "thread_summary"}

PATH_DEADLINE_FIELD = {
    "rag": "deadline_seconds_standard",
    "nl2sql": "deadline_seconds_standard",
    "hybrid": "deadline_seconds_hybrid",
    "agentic": "deadline_seconds_agentic",
    "high_risk_panel": "deadline_seconds_high_risk",
}

RELEASE_TIER_NODES = {"output_guardrail", "escalation_manager", "safe_refusal", "clarification"}

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

    def check(self, node_name: str) -> BudgetVerdict:
        if node_name in RELEASE_TIER_NODES:
            return BudgetVerdict(allowed=True)

        estimate = NODE_TOKEN_ESTIMATE.get(node_name, 500)
        optional = node_name in OPTIONAL_TIER_NODES

        if self.seconds_remaining <= 0:
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
    widened = widened_deadline(state, path_deadline_seconds(path))

    if widened is None:
        return {}

    return {"deadline_ts": widened}
