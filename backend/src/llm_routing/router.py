"""Tier selection per node: which model answers, and why.

Three node classes:

* **fixed small** - rewrite, intent, risk classifier, planner, template
  selector, SQL narration, reflection. These are short structured-output
  calls the small model handles well; sending them to the strong model buys
  nothing.
* **fixed strong** - the four high-risk panel agents. A High-risk answer
  always gets the strongest reasoning available; the router never trades
  that for cost.
* **routable** - ``rag_generate``, ``hybrid_generate``,
  ``compliance_validation`` and ``agentic_rag``. These write or judge the
  answer and are the expensive calls. Here the strategy decides:

  ==============  =====================================================
  static          the tier in ``configs.llms.TIER_FOR_NODE`` (strong)
  heuristic       :func:`assess_complexity` picks small for simple
                  questions and strong for complex ones (default)
  cost_saver      small unless the fused risk is High
  quality_first   strong, and never downgraded for budget pressure
  ==============  =====================================================

Budget pressure (latency): under ``heuristic``, ``cost_saver`` and ``static``,
when the request has less than ``ROUTING_LATENCY_PRESSURE_SECONDS`` of
deadline or ``ROUTING_TOKEN_PRESSURE`` tokens left, a routable node drops to
the small tier so the request finishes inside its SLO instead of escalating
on a budget stop. A High-risk request is exempt: correctness outranks speed
there, and the panel path already carries the widest deadline.

Every decision is returned as a :class:`RoutingDecision` and appended to the
graph state's ``model_routing`` channel by the node, so it appears in the
``/ask`` trace, the reviewer package and the audit log.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any

from configs.settings import settings
from src.llm_routing.complexity import ComplexityAssessment, assess_complexity

logger = logging.getLogger(__name__)

SMALL = "small"
STRONG = "strong"

FIXED_SMALL = frozenset(
    {
        "query_rewrite",
        "thread_summary",
        "intent_classification",
        "entity_resolution",
        "risk_l2_classifier",
        "planner",
        "sql_template_selector",
        "nl2sql_intent",
        "sql_narration",
        "reflection",
        "sql_intent_guard",
        "nl2sql_generate",
        "table_summary",
        "metadata_extraction",
    }
)

FIXED_STRONG = frozenset(
    {
        "panel_policy_interpreter",
        "panel_data_verifier",
        "panel_challenger",
        "panel_consensus",
        "panel_repair",
    }
)

ROUTABLE = frozenset({"rag_generate", "hybrid_generate", "compliance_validation", "agentic_rag"})

DEFAULT_ROUTABLE_TIER = STRONG


@dataclass(slots=True)
class RoutingDecision:
    node: str
    tier: str
    strategy: str
    reason: str
    routable: bool = False
    budget_pressure: bool = False
    seconds_remaining: float | None = None
    tokens_remaining: int | None = None
    assessment: ComplexityAssessment | None = None
    downgraded: bool = False

    @property
    def deployment(self) -> str:
        return (
            settings.azure_openai_strong_deployment
            if self.tier == STRONG
            else settings.azure_openai_small_deployment
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "node": self.node,
            "tier": self.tier,
            "deployment": self.deployment,
            "strategy": self.strategy,
            "reason": self.reason,
            "routable": self.routable,
            "budget_pressure": self.budget_pressure,
            "downgraded": self.downgraded,
        }

        if self.seconds_remaining is not None:
            payload["seconds_remaining"] = round(self.seconds_remaining, 2)

        if self.tokens_remaining is not None:
            payload["tokens_remaining"] = self.tokens_remaining

        if self.assessment is not None:
            payload["complexity"] = {
                "label": self.assessment.label,
                "score": self.assessment.score,
                "reasons": list(self.assessment.reasons),
            }

        return payload


class RoutingStats:
    """Counts of decisions, for ``GET /metrics/optimization``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.by_tier: dict[str, int] = {SMALL: 0, STRONG: 0}
        self.by_node: dict[str, dict[str, int]] = {}
        self.downgrades: int = 0
        self.budget_pressure: int = 0
        self.routable_decisions: int = 0
        self.routable_small: int = 0

    def record(self, decision: RoutingDecision) -> None:
        with self._lock:
            self.by_tier[decision.tier] = self.by_tier.get(decision.tier, 0) + 1
            node = self.by_node.setdefault(decision.node, {SMALL: 0, STRONG: 0})
            node[decision.tier] = node.get(decision.tier, 0) + 1

            if decision.routable:
                self.routable_decisions += 1

                if decision.tier == SMALL:
                    self.routable_small += 1

            if decision.downgraded:
                self.downgrades += 1

            if decision.budget_pressure:
                self.budget_pressure += 1

    def report(self) -> dict[str, Any]:
        with self._lock:
            routable = self.routable_decisions

            return {
                "strategy": settings.model_routing_strategy,
                "decisions_by_tier": dict(self.by_tier),
                "decisions_by_node": {name: dict(counts) for name, counts in self.by_node.items()},
                "routable_decisions": routable,
                "routable_small_share": round(self.routable_small / routable, 4) if routable else 0.0,
                "downgrades": self.downgrades,
                "budget_pressure_events": self.budget_pressure,
                "pressure_thresholds": {
                    "seconds": settings.routing_latency_pressure_seconds,
                    "tokens": settings.routing_token_pressure,
                },
            }

    def reset(self) -> None:
        with self._lock:
            self.by_tier = {SMALL: 0, STRONG: 0}
            self.by_node = {}
            self.downgrades = 0
            self.budget_pressure = 0
            self.routable_decisions = 0
            self.routable_small = 0


routing_stats = RoutingStats()


def _context_from_state(state: dict | None) -> dict[str, Any]:
    if not state:
        return {}

    risk = state.get("risk")
    intent = state.get("intent")

    return {
        "query": state.get("standalone_query")
        or state.get("sanitised_query")
        or state.get("raw_query")
        or "",
        "risk_level": risk.final_level.value if risk is not None else None,
        "intent": intent.intent.value if intent is not None else None,
        "evidence_path": state.get("routed_path") or state.get("evidence_path"),
        "entity_count": len(state.get("resolved_entities") or []),
        "history_turns": len(state.get("conversation_history") or []),
    }


def _budget(state: dict | None) -> tuple[float | None, int | None]:
    if not state or "deadline_ts" not in state:
        return None, None

    from src.core.budget import guard_from_state

    guard = guard_from_state(state)

    return guard.seconds_remaining, guard.tokens_remaining


def route_tier(
    node_name: str,
    state: dict | None = None,
    *,
    strategy: str | None = None,
    query: str | None = None,
) -> RoutingDecision:
    """Pick the tier for ``node_name`` given the graph state (or a bare query)."""
    strategy = (strategy or settings.model_routing_strategy or "heuristic").lower()

    if node_name in FIXED_STRONG:
        decision = RoutingDecision(
            node=node_name,
            tier=STRONG,
            strategy=strategy,
            reason="high-risk panel agents always use the strong tier",
        )
        routing_stats.record(decision)

        return decision

    if node_name in FIXED_SMALL or node_name not in ROUTABLE:
        decision = RoutingDecision(
            node=node_name,
            tier=SMALL,
            strategy=strategy,
            reason="structured classification / narration call; the small tier is sufficient",
        )
        routing_stats.record(decision)

        return decision

    context = _context_from_state(state)

    if query is not None:
        context["query"] = query

    seconds_remaining, tokens_remaining = _budget(state)
    risk_high = (context.get("risk_level") or "").lower() == "high"

    pressure = bool(
        (seconds_remaining is not None and seconds_remaining < settings.routing_latency_pressure_seconds)
        or (tokens_remaining is not None and tokens_remaining < settings.routing_token_pressure)
    )

    assessment: ComplexityAssessment | None = None

    if strategy == "quality_first":
        tier, reason = STRONG, "quality_first strategy: routable nodes always use the strong tier"
    elif strategy == "cost_saver":
        if risk_high:
            tier, reason = STRONG, "cost_saver strategy, but risk is High so the strong tier is kept"
        else:
            tier, reason = SMALL, "cost_saver strategy: routable nodes use the small tier"
    elif strategy == "static":
        tier, reason = DEFAULT_ROUTABLE_TIER, "static strategy: fixed TIER_FOR_NODE table"
    else:
        assessment = assess_complexity(
            context.get("query", ""),
            risk_level=context.get("risk_level"),
            intent=context.get("intent"),
            evidence_path=context.get("evidence_path"),
            entity_count=int(context.get("entity_count", 0)),
            history_turns=int(context.get("history_turns", 0)),
        )
        tier = assessment.tier
        reason = f"heuristic assessment: {assessment.label} (score {assessment.score}; {'; '.join(assessment.reasons[:3])})"

    downgraded = False

    if pressure and tier == STRONG and strategy != "quality_first" and not risk_high:
        tier = SMALL
        downgraded = True
        reason = (
            f"budget pressure ({_fmt_seconds(seconds_remaining)}s and {tokens_remaining} tokens left) "
            f"forces the small tier so the request finishes inside its deadline; original reason: {reason}"
        )

    decision = RoutingDecision(
        node=node_name,
        tier=tier,
        strategy=strategy,
        reason=reason,
        routable=True,
        budget_pressure=pressure,
        seconds_remaining=seconds_remaining,
        tokens_remaining=tokens_remaining,
        assessment=assessment,
        downgraded=downgraded,
    )

    routing_stats.record(decision)

    logger.info(
        "model_route node=%s tier=%s deployment=%s strategy=%s pressure=%s downgraded=%s reason=%s",
        node_name,
        tier,
        decision.deployment,
        strategy,
        pressure,
        downgraded,
        reason,
    )

    return decision


def _fmt_seconds(value: float | None) -> str:
    return "?" if value is None else f"{value:.1f}"


def routing_report() -> dict[str, Any]:
    from src.llm_routing.pricing import tier_prices

    return {
        **routing_stats.report(),
        "fixed_small": sorted(FIXED_SMALL),
        "fixed_strong": sorted(FIXED_STRONG),
        "routable": sorted(ROUTABLE),
        "tiers": tier_prices(),
        "gateway": settings.llm_gateway_url or None,
    }


__all__ = [
    "FIXED_SMALL",
    "FIXED_STRONG",
    "ROUTABLE",
    "SMALL",
    "STRONG",
    "RoutingDecision",
    "RoutingStats",
    "route_tier",
    "routing_report",
    "routing_stats",
]
