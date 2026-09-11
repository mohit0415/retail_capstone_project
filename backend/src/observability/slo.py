import logging

from configs.database import read_only_connection, writable_connection
from configs.settings import settings
from src.core.budget import path_deadline_seconds

logger = logging.getLogger(__name__)

STAGE_OF_NODE = {
    "input_guardrail": "t1",
    "query_rewrite": "t1",
    "intent_classification": "t1",
    "entity_resolution": "t1",
    "risk_assessment": "t1",
    "planner": "t2",
    "rag_path": "t3",
    "nl2sql_path": "t3",
    "hybrid_path": "t3",
    "agentic_rag": "t3",
    "multi_agent_panel": "t3",
    "compliance_validation": "t4",
    "reflection": "t4",
    "confidence_scoring": "t4",
    "output_guardrail": "t4",
    "escalation_manager": "t4",
    "safe_refusal": "t4",
    "clarification": "t4",
    "no_answer": "t4",
}

STAGE_ORDER = ["t1", "t2", "t3", "t4"]

STAGE_LABELS = {
    "t1": "intake complete (guardrail, rewrite, intent, entities, risk)",
    "t2": "evidence plan ready",
    "t3": "evidence gathered and answer drafted",
    "t4": "validated, scored and released",
}

STAGE_TARGETS = {
    "t1": "slo_t1_p95_ms",
    "t2": "slo_t2_p95_ms",
    "t3": "slo_t3_p95_ms",
    "t4": "slo_t4_p95_ms",
}

INSERT_LATENCY = """
INSERT INTO request_latency (
    request_id, thread_id, role, outcome, evidence_path, risk_level,
    degraded, t1_ms, t2_ms, t3_ms, t4_ms, total_ms
)
VALUES (
    %(request_id)s, %(thread_id)s, %(role)s, %(outcome)s, %(evidence_path)s, %(risk_level)s,
    %(degraded)s, %(t1_ms)s, %(t2_ms)s, %(t3_ms)s, %(t4_ms)s, %(total_ms)s
)
ON CONFLICT DO NOTHING
"""

PERCENTILES = """
SELECT
    COUNT(*) AS sample_size,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY t1_ms) AS t1_p50,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY t1_ms) AS t1_p95,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY t1_ms) AS t1_p99,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY t2_ms) AS t2_p50,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY t2_ms) AS t2_p95,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY t2_ms) AS t2_p99,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY t3_ms) AS t3_p50,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY t3_ms) AS t3_p95,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY t3_ms) AS t3_p99,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY t4_ms) AS t4_p50,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY t4_ms) AS t4_p95,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY t4_ms) AS t4_p99,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY total_ms) AS total_p50,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY total_ms) AS total_p95,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY total_ms) AS total_p99
FROM request_latency
WHERE created_at >= now() - make_interval(hours => %(hours)s)
"""

OUTCOME_MIX = """
SELECT outcome, COUNT(*) AS hits
FROM request_latency
WHERE created_at >= now() - make_interval(hours => %(hours)s)
GROUP BY outcome
ORDER BY hits DESC
"""

PATH_MIX = """
SELECT
    COALESCE(evidence_path, 'none') AS evidence_path,
    COUNT(*) AS hits,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY total_ms) AS total_p95
FROM request_latency
WHERE created_at >= now() - make_interval(hours => %(hours)s)
GROUP BY 1
ORDER BY hits DESC
"""


def stage_for(node_name: str) -> str | None:
    return STAGE_OF_NODE.get(node_name)


def stage_marks(marks: list[dict]) -> dict[str, float]:
    reached: dict[str, float] = {}

    for mark in marks or []:
        stage = mark.get("stage")
        elapsed = mark.get("elapsed_ms")

        if stage is None or elapsed is None:
            continue

        if stage not in reached or elapsed > reached[stage]:
            reached[stage] = float(elapsed)

    carried = None

    for stage in STAGE_ORDER:
        if stage in reached:
            carried = reached[stage]
        elif carried is not None:
            reached[stage] = carried

    return reached


def record_latency(state: dict, total_ms: float, outcome: str) -> None:
    reached = stage_marks(state.get("marks", []))
    risk = state.get("risk")

    payload = {
        "request_id": state.get("request_id", "unknown"),
        "thread_id": state.get("thread_id", "unknown"),
        "role": state.get("role", "unknown"),
        "outcome": outcome,
        "evidence_path": state.get("evidence_path"),
        "risk_level": risk.final_level.value if risk else None,
        "degraded": bool(state.get("degraded", False)),
        "t1_ms": reached.get("t1"),
        "t2_ms": reached.get("t2"),
        "t3_ms": reached.get("t3"),
        "t4_ms": reached.get("t4"),
        "total_ms": round(total_ms, 2),
    }

    try:
        with writable_connection() as conn:
            conn.execute(INSERT_LATENCY, payload)
    except Exception as exc:
        logger.error("latency write failed request_id=%s error=%s", payload["request_id"], exc)


def _target_for(stage: str) -> float:
    return float(getattr(settings, STAGE_TARGETS[stage]))


def path_target_ms(evidence_path: str | None) -> float:
    if not evidence_path or evidence_path == "none":
        return float(settings.slo_total_p95_ms)

    return round(path_deadline_seconds(evidence_path) * 1000 * settings.slo_path_target_ratio, 2)


def latency_report(hours: int = 24) -> dict:
    with read_only_connection() as conn:
        summary = conn.execute(PERCENTILES, {"hours": hours}).fetchone()
        outcomes = conn.execute(OUTCOME_MIX, {"hours": hours}).fetchall()
        paths = conn.execute(PATH_MIX, {"hours": hours}).fetchall()

    sample_size = int(summary["sample_size"]) if summary else 0

    stages = []
    breached = []

    for stage in STAGE_ORDER:
        target = _target_for(stage)
        p95 = summary[f"{stage}_p95"] if summary else None
        meets = None

        if p95 is not None:
            meets = float(p95) <= target

            if not meets:
                breached.append(stage)

        stages.append(
            {
                "stage": stage,
                "meaning": STAGE_LABELS[stage],
                "p50_ms": _round(summary[f"{stage}_p50"]) if summary else None,
                "p95_ms": _round(p95),
                "p99_ms": _round(summary[f"{stage}_p99"]) if summary else None,
                "target_p95_ms": target,
                "meets_slo": meets,
            }
        )

    path_rows = []
    breached_paths = []
    weighted_target = 0.0
    weighted_count = 0

    for row in paths:
        name = row["evidence_path"]
        target = path_target_ms(name)
        hits = int(row["hits"])
        p95 = row["total_p95"]
        meets = None

        if p95 is not None:
            meets = float(p95) <= target

            if not meets:
                breached_paths.append(name)

        weighted_target += target * hits
        weighted_count += hits

        path_rows.append(
            {
                "evidence_path": name,
                "count": hits,
                "total_p95_ms": _round(p95),
                "target_p95_ms": target,
                "meets_slo": meets,
            }
        )

    if weighted_count:
        total_target = round(weighted_target / weighted_count, 2)
    else:
        total_target = float(settings.slo_total_p95_ms)

    total_p95 = summary["total_p95"] if summary else None
    total_meets = None

    if total_p95 is not None:
        total_meets = float(total_p95) <= total_target

        if not total_meets:
            breached.append("total")

    return {
        "window_hours": hours,
        "sample_size": sample_size,
        "measured_from": "t0, the moment /ask accepted the request",
        "total_target_basis": (
            "the sample-weighted mean of the per-path targets seen in this window, so a window "
            "holding high-risk requests is not judged against the standard-path objective"
        ),
        "stages": stages,
        "total": {
            "p50_ms": _round(summary["total_p50"]) if summary else None,
            "p95_ms": _round(total_p95),
            "p99_ms": _round(summary["total_p99"]) if summary else None,
            "target_p95_ms": total_target,
            "meets_slo": total_meets,
        },
        "breached_stages": breached,
        "breached_paths": breached_paths,
        "outcomes": [{"outcome": row["outcome"], "count": int(row["hits"])} for row in outcomes],
        "evidence_paths": path_rows,
    }


def _round(value) -> float | None:
    if value is None:
        return None

    return round(float(value), 2)
