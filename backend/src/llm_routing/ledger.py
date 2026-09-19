"""Cost ledger: real token usage and spend, per request and process-wide.

The graph's ``tokens_spent`` channel is a *budget estimate* the budget guard
uses before a node runs. This ledger records what the provider actually
billed, taken from the token usage LangChain hands back on every call (see
:class:`src.observability.callbacks.LoggingCallbackHandler`), priced with
:mod:`src.llm_routing.pricing`.

Per-request rows are kept in a bounded LRU so a long-running server does not
grow without bound; the process totals are cumulative since start-up. A
request's summary is attached to its ``/ask`` trace as ``llm_usage`` and the
totals are rendered by ``GET /metrics/optimization``.
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from configs.settings import settings
from src.llm_routing.pricing import estimate_cost, price_for

logger = logging.getLogger(__name__)

MAX_TRACKED_REQUESTS = 2000


@dataclass(slots=True)
class ModelUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    elapsed_ms: float = 0.0
    cache_hits: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "usd": round(self.usd, 6),
            "elapsed_ms": round(self.elapsed_ms, 1),
            "cache_hits": self.cache_hits,
        }


@dataclass(slots=True)
class RequestUsage:
    request_id: str
    started_at: float = field(default_factory=time.time)
    total: ModelUsage = field(default_factory=ModelUsage)
    by_model: dict[str, ModelUsage] = field(default_factory=dict)
    by_node: dict[str, ModelUsage] = field(default_factory=dict)
    budget_warned: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            **self.total.as_dict(),
            "by_model": {name: usage.as_dict() for name, usage in self.by_model.items()},
            "by_node": {name: usage.as_dict() for name, usage in self.by_node.items()},
            "over_budget": self.total.usd > settings.cost_budget_usd_per_request,
            "budget_usd": settings.cost_budget_usd_per_request,
        }


class CostLedger:
    def __init__(self, max_requests: int = MAX_TRACKED_REQUESTS):
        self._lock = threading.Lock()
        self._requests: OrderedDict[str, RequestUsage] = OrderedDict()
        self._max_requests = max_requests
        self.process = ModelUsage()
        self.by_model: dict[str, ModelUsage] = {}
        self.by_tier: dict[str, ModelUsage] = {
            "small": ModelUsage(),
            "strong": ModelUsage(),
            "other": ModelUsage(),
        }
        self.started_at = time.time()

    def _request(self, request_id: str | None) -> RequestUsage:
        key = request_id or "-"
        usage = self._requests.get(key)

        if usage is None:
            usage = RequestUsage(request_id=key)
            self._requests[key] = usage

            while len(self._requests) > self._max_requests:
                self._requests.popitem(last=False)
        else:
            self._requests.move_to_end(key)

        return usage

    @staticmethod
    def _bump(target: ModelUsage, prompt: int, completion: int, usd: float, elapsed_ms: float) -> None:
        target.calls += 1
        target.prompt_tokens += prompt
        target.completion_tokens += completion
        target.usd += usd
        target.elapsed_ms += elapsed_ms

    def record_call(
        self,
        request_id: str | None,
        *,
        node: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        elapsed_ms: float | None = None,
    ) -> float:
        """Record one billed model call; returns its USD cost."""
        prompt = int(prompt_tokens or 0)
        completion = int(completion_tokens or 0)
        usd = estimate_cost(model, prompt, completion)
        elapsed = float(elapsed_ms or 0.0)
        tier = tier_of(model)

        with self._lock:
            request = self._request(request_id)
            self._bump(request.total, prompt, completion, usd, elapsed)
            self._bump(request.by_model.setdefault(model, ModelUsage()), prompt, completion, usd, elapsed)
            self._bump(request.by_node.setdefault(node, ModelUsage()), prompt, completion, usd, elapsed)
            self._bump(self.process, prompt, completion, usd, elapsed)
            self._bump(self.by_model.setdefault(model, ModelUsage()), prompt, completion, usd, elapsed)
            self._bump(self.by_tier[tier], prompt, completion, usd, elapsed)

            over_budget = (
                request.total.usd > settings.cost_budget_usd_per_request and not request.budget_warned
            )

            if over_budget:
                request.budget_warned = True

            running_usd = request.total.usd

        logger.debug(
            "cost node=%s model=%s tier=%s tokens=%d/%d usd=%.6f request_usd=%.6f",
            node,
            model,
            tier,
            prompt,
            completion,
            usd,
            running_usd,
        )

        if over_budget:
            logger.warning(
                "cost budget exceeded request_id=%s spent_usd=%.4f budget_usd=%.4f (soft limit, request continues)",
                request_id,
                running_usd,
                settings.cost_budget_usd_per_request,
            )

        return usd

    def record_cache_hit(self, request_id: str | None, model: str = "unknown") -> None:
        with self._lock:
            request = self._request(request_id)
            request.total.cache_hits += 1
            request.by_model.setdefault(model, ModelUsage()).cache_hits += 1
            self.process.cache_hits += 1
            self.by_model.setdefault(model, ModelUsage()).cache_hits += 1
            self.by_tier[tier_of(model)].cache_hits += 1

    def summary(self, request_id: str | None) -> dict[str, Any]:
        with self._lock:
            request = self._requests.get(request_id or "-")

            if request is None:
                return RequestUsage(request_id=request_id or "-").as_dict()

            return request.as_dict()

    def forget(self, request_id: str) -> None:
        with self._lock:
            self._requests.pop(request_id, None)

    def report(self) -> dict[str, Any]:
        with self._lock:
            tracked = len(self._requests)
            per_request = [r.total.usd for r in self._requests.values()]
            per_request_tokens = [
                r.total.prompt_tokens + r.total.completion_tokens for r in self._requests.values()
            ]
            process = self.process.as_dict()
            by_model = {name: usage.as_dict() for name, usage in self.by_model.items()}
            by_tier = {name: usage.as_dict() for name, usage in self.by_tier.items()}

        mean = round(sum(per_request) / len(per_request), 6) if per_request else 0.0
        peak = round(max(per_request), 6) if per_request else 0.0
        over = sum(1 for value in per_request if value > settings.cost_budget_usd_per_request)

        return {
            "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started_at)),
            "process": process,
            "by_model": by_model,
            "by_tier": by_tier,
            "requests_tracked": tracked,
            "usd_per_request_mean": mean,
            "usd_per_request_peak": peak,
            "tokens_per_request_mean": round(sum(per_request_tokens) / len(per_request_tokens))
            if per_request_tokens
            else 0,
            "tokens_per_request_max": max(per_request_tokens) if per_request_tokens else 0,
            "requests_over_budget": over,
            "budget_usd_per_request": settings.cost_budget_usd_per_request,
            "small_tier_share": _share(by_tier, "small"),
        }

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()
            self.process = ModelUsage()
            self.by_model = {}
            self.by_tier = {"small": ModelUsage(), "strong": ModelUsage(), "other": ModelUsage()}
            self.started_at = time.time()


def tier_of(model: str | None) -> str:
    """Which configured tier a model / deployment name belongs to."""
    name = (model or "").lower()
    small = settings.azure_openai_small_deployment.lower()
    strong = settings.azure_openai_strong_deployment.lower()

    if name in (small, settings.llm_gateway_small_model.lower()) or (small and name.startswith(small)):
        return "small"

    if name in (strong, settings.llm_gateway_strong_model.lower()) or (strong and name.startswith(strong)):
        return "strong"

    resolved = price_for(name)

    if resolved.input_per_million == 0.0 and resolved.output_per_million == 0.0:
        return "other"

    return "small" if resolved.input_per_million <= price_for(small).input_per_million else "strong"


def _share(by_tier: dict[str, dict[str, Any]], tier: str) -> float:
    total = sum(int(row.get("calls", 0)) for row in by_tier.values())

    return round(int(by_tier.get(tier, {}).get("calls", 0)) / total, 4) if total else 0.0


cost_ledger = CostLedger()
