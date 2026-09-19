"""Runtime latency percentiles pulled from the Langfuse portal.

The local ``request_latency`` table measures wall-clock time inside ``/ask``.
Langfuse traces the same requests from the outside (every graph run is one
``policy_graph`` trace), so its p50/p95 is the independent, portal-side view
of the same SLO. ``GET /metrics/langfuse`` serves this to the SLO page next
to the database-backed numbers.

Uses the Langfuse Metrics API (``GET /api/public/metrics``) with the same
keys the tracer runs on. Rows come back keyed ``{aggregation}_{measure}``
(``p95_latency``); the traces view reports latency in milliseconds.

The same query also brings back token consumption (p95 and max total tokens
per trace - the "cost in tokens" view of the window) and the summed USD cost
Langfuse computed for those traces.
"""

import json
import logging
from datetime import UTC, datetime, timedelta

from configs.settings import settings

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 12.0

TRACE_NAME = "policy_graph"


def _row_value(row: dict, measure: str, aggregation: str) -> float | None:
    """Read one metric from a data row, tolerating either key order."""
    for key in (f"{aggregation}_{measure}", f"{measure}_{aggregation}"):
        value = row.get(key)

        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

    return None


def _as_ms(value: float | None) -> float | None:
    if value is None:
        return None

    # the traces view reports milliseconds; a value this small for an LLM graph
    # can only mean a deployment that reports seconds - normalise it
    if 0 < value < 30:
        value *= 1000.0

    return round(value, 2)


def latency_percentiles(hours: int = 24) -> dict:
    """p50/p95 trace latency for the window, or ``{"enabled": False}`` without keys."""
    public_key = settings.langfuse_public_key
    secret_key = settings.langfuse_secret_key
    host = (settings.langfuse_host or "https://cloud.langfuse.com").rstrip("/")

    base = {
        "enabled": bool(public_key and secret_key),
        "host": host,
        "window_hours": hours,
        "trace_name": TRACE_NAME,
        "trace_count": None,
        "p50_ms": None,
        "p95_ms": None,
        "total_tokens_p95": None,
        "total_tokens_max": None,
        "total_cost_usd": None,
        "error": None,
    }

    if not base["enabled"]:
        base["error"] = "langfuse keys are not configured"

        return base

    now = datetime.now(UTC)
    query = {
        "view": "traces",
        "metrics": [
            {"measure": "latency", "aggregation": "p50"},
            {"measure": "latency", "aggregation": "p95"},
            {"measure": "count", "aggregation": "count"},
            {"measure": "totalTokens", "aggregation": "p95"},
            {"measure": "totalTokens", "aggregation": "max"},
            {"measure": "totalCost", "aggregation": "sum"},
        ],
        "dimensions": [],
        "filters": [
            {"column": "name", "operator": "=", "value": TRACE_NAME, "type": "string"},
        ],
        "fromTimestamp": (now - timedelta(hours=hours)).isoformat().replace("+00:00", "Z"),
        "toTimestamp": now.isoformat().replace("+00:00", "Z"),
    }

    import httpx

    try:
        response = httpx.get(
            f"{host}/api/public/metrics",
            params={"query": json.dumps(query)},
            auth=(public_key, secret_key),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        rows = response.json().get("data") or []
    except Exception as exc:
        logger.warning("langfuse metrics query failed: %s", str(exc)[:200])
        base["error"] = str(exc)[:200]

        return base

    if not rows:
        return base

    row = rows[0]
    count = _row_value(row, "count", "count")

    base["trace_count"] = int(count) if count is not None else None
    base["p50_ms"] = _as_ms(_row_value(row, "latency", "p50"))
    base["p95_ms"] = _as_ms(_row_value(row, "latency", "p95"))

    tokens_p95 = _row_value(row, "totalTokens", "p95")
    tokens_max = _row_value(row, "totalTokens", "max")
    cost_sum = _row_value(row, "totalCost", "sum")

    base["total_tokens_p95"] = round(tokens_p95) if tokens_p95 is not None else None
    base["total_tokens_max"] = round(tokens_max) if tokens_max is not None else None
    base["total_cost_usd"] = round(cost_sum, 6) if cost_sum is not None else None

    return base
