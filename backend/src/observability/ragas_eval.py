"""Per-request RAGAS answer-quality scoring.

Every certified answer is scored in the background (``BackgroundTasks`` after
``/ask`` has already responded) on four RAGAS metrics:

* ``faithfulness`` - are the claims in the answer supported by the retrieved
  clauses;
* ``answer_accuracy`` - a binary RAGAS ``AspectCritic`` verdict: does the
  answer correctly answer the question with every factual statement agreeing
  with the retrieved clauses. The rolling mean is the accuracy rate;
* ``context_precision`` - are the retrieved clauses actually relevant to the
  answer (``LLMContextPrecisionWithoutReference``: no golden answer needed);
* ``context_recall`` - did retrieval bring back the clauses the answer needed.
  True recall needs a ground-truth reference, which a live request does not
  have, so the generated answer stands in as the reference. The UI labels it
  as a proxy; the golden-set eval (``evals/run_eval.py``) remains the place
  where recall against real references is measured.

The judge is the small-tier chat model (the same per-user Azure credentials
the graph runs on), so scoring works with whatever the login page supplied.
Scores land in ``request_evaluations`` keyed by ``request_id``; the frontend
fetches them lazily via ``GET /requests/{request_id}/evaluation`` and the SLO
page aggregates them via ``GET /metrics/ragas``. A scoring failure only ever
logs - it must never affect the request that produced the answer.
"""

import asyncio
import logging
import math
import re
import time

from configs.database import read_only_connection, writable_connection
from configs.llms import ModelTier, deployment_for, model_for_tier

logger = logging.getLogger(__name__)

# one judge call chain per metric; a hung judge should not pin a worker forever
METRIC_TIMEOUT_SECONDS = 90.0

# rolling-window quality objectives shown on the SLO page
QUALITY_TARGETS = {
    "faithfulness": 0.85,
    "answer_accuracy": 0.90,
    "context_precision": 0.75,
    "context_recall": 0.80,
}

# the AspectCritic judge answers this yes/no question per answer; the rolling
# mean of the 0/1 verdicts is the accuracy rate shown on the SLO page
ACCURACY_DEFINITION = (
    "Does the response correctly and accurately answer the user's question, "
    "with every factual statement supported by (and consistent with) the "
    "retrieved contexts? Answer no if any figure, condition or claim is wrong, "
    "contradicted or unsupported."
)

# answers under this faithfulness are counted as "low faithfulness" in the report
LOW_FAITHFULNESS_CEILING = 0.70

INSERT_EVALUATION = """
INSERT INTO request_evaluations (
    request_id, thread_id, evidence_path, status, skipped_reason,
    faithfulness, answer_accuracy, context_precision, context_recall,
    judge_model, contexts_scored, duration_ms, error
)
VALUES (
    %(request_id)s, %(thread_id)s, %(evidence_path)s, %(status)s, %(skipped_reason)s,
    %(faithfulness)s, %(answer_accuracy)s, %(context_precision)s, %(context_recall)s,
    %(judge_model)s, %(contexts_scored)s, %(duration_ms)s, %(error)s
)
ON CONFLICT (request_id) DO NOTHING
"""

SELECT_EVALUATION = """
SELECT request_id, thread_id, evidence_path, status, skipped_reason,
       faithfulness, answer_accuracy, context_precision, context_recall,
       judge_model, contexts_scored, duration_ms, error, created_at
FROM request_evaluations
WHERE request_id = %(request_id)s
"""

QUALITY_SUMMARY = """
SELECT
    COUNT(*) FILTER (WHERE status = 'done')    AS scored,
    COUNT(*) FILTER (WHERE status = 'skipped') AS skipped,
    COUNT(*) FILTER (WHERE status = 'error')   AS failed,
    AVG(faithfulness)                          AS faithfulness_mean,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY faithfulness)      AS faithfulness_p50,
    AVG(answer_accuracy)                       AS answer_accuracy_mean,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY answer_accuracy)   AS answer_accuracy_p50,
    AVG(context_precision)                     AS context_precision_mean,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY context_precision) AS context_precision_p50,
    AVG(context_recall)                        AS context_recall_mean,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY context_recall)    AS context_recall_p50,
    COUNT(*) FILTER (WHERE faithfulness < %(low_ceiling)s)          AS low_faithfulness
FROM request_evaluations
WHERE created_at >= now() - make_interval(hours => %(hours)s)
"""

RECENT_EVALUATIONS = """
SELECT request_id, evidence_path, status, faithfulness, answer_accuracy, context_precision, context_recall, created_at
FROM request_evaluations
WHERE created_at >= now() - make_interval(hours => %(hours)s)
ORDER BY created_at DESC
LIMIT %(limit)s
"""


def _clean(score) -> float | None:
    """RAGAS answers ``nan`` when the judge output could not be parsed."""
    if score is None:
        return None

    value = float(score)

    if math.isnan(value) or math.isinf(value):
        return None

    return round(min(1.0, max(0.0, value)), 4)


async def _score_metrics(question: str, answer: str, contexts: list[str]) -> dict[str, float | None]:
    from ragas import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AspectCritic,
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
    )

    judge = LangchainLLMWrapper(model_for_tier(ModelTier.SMALL, temperature=0.0))

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        # recall proxy: the certified answer stands in for the missing golden reference
        reference=answer,
    )

    metrics = {
        "faithfulness": Faithfulness(llm=judge),
        "answer_accuracy": AspectCritic(name="answer_accuracy", definition=ACCURACY_DEFINITION, llm=judge),
        "context_precision": LLMContextPrecisionWithoutReference(llm=judge),
        "context_recall": LLMContextRecall(llm=judge),
    }

    scores: dict[str, float | None] = {}

    for name, metric in metrics.items():
        try:
            raw = await asyncio.wait_for(metric.single_turn_ascore(sample), timeout=METRIC_TIMEOUT_SECONDS)
            scores[name] = _clean(raw)
        except Exception as exc:
            logger.warning("ragas metric %s failed: %s", name, str(exc)[:200])
            scores[name] = None

    return scores


def _store(payload: dict) -> None:
    try:
        with writable_connection() as conn:
            conn.execute(INSERT_EVALUATION, payload)
    except Exception as exc:
        logger.error("evaluation write failed request_id=%s error=%s", payload["request_id"], exc)


async def evaluate_answer(
    request_id: str,
    thread_id: str,
    question: str,
    answer: str,
    contexts: list[str],
    evidence_path: str | None,
) -> None:
    """Score one certified answer and store the result. Never raises."""
    started = time.monotonic()

    base = {
        "request_id": request_id,
        "thread_id": thread_id,
        "evidence_path": evidence_path,
        "status": "done",
        "skipped_reason": None,
        "faithfulness": None,
        "answer_accuracy": None,
        "context_precision": None,
        "context_recall": None,
        "judge_model": deployment_for(ModelTier.SMALL),
        "contexts_scored": len(contexts),
        "duration_ms": None,
        "error": None,
    }

    # ingestion keeps hard line breaks mid-sentence ("1\nCustomer transaction
    # records: ..."), which measurably destabilises the judge's NLI verdicts -
    # collapse whitespace before scoring
    contexts = [re.sub(r"\s+", " ", c).strip() for c in contexts if c and c.strip()]

    if not contexts:
        # a pure record answer (nl2sql with no retrieved clauses) has nothing to ground against
        base.update(status="skipped", skipped_reason="no_retrieved_contexts", contexts_scored=0)
        _store(base)

        return

    if not answer.strip():
        base.update(status="skipped", skipped_reason="empty_answer")
        _store(base)

        return

    try:
        scores = await _score_metrics(question, answer, contexts)
    except Exception as exc:
        logger.error("ragas evaluation failed request_id=%s: %s", request_id, str(exc)[:300])
        base.update(
            status="error",
            error=str(exc)[:300],
            duration_ms=round((time.monotonic() - started) * 1000, 2),
        )
        _store(base)

        return

    base.update(scores)
    base["duration_ms"] = round((time.monotonic() - started) * 1000, 2)

    if all(scores[name] is None for name in scores):
        base.update(status="error", error="every metric returned unparseable output")

    _store(base)

    logger.info(
        "ragas scored request_id=%s faithfulness=%s accuracy=%s precision=%s recall=%s contexts=%d in %.0f ms",
        request_id,
        base["faithfulness"],
        base["answer_accuracy"],
        base["context_precision"],
        base["context_recall"],
        len(contexts),
        base["duration_ms"],
    )


def fetch_evaluation(request_id: str) -> dict | None:
    try:
        with read_only_connection() as conn:
            row = conn.execute(SELECT_EVALUATION, {"request_id": request_id}).fetchone()
    except Exception as exc:
        logger.error("evaluation read failed request_id=%s error=%s", request_id, exc)

        return None

    return dict(row) if row else None


def _metric_row(name: str, mean, p50, scored: int) -> dict:
    target = QUALITY_TARGETS[name]
    mean = _clean(mean)

    return {
        "metric": name,
        "mean": mean,
        "p50": _clean(p50),
        "target_mean": target,
        "meets_target": (mean >= target) if (mean is not None and scored > 0) else None,
    }


def quality_report(hours: int = 24, recent_limit: int = 20) -> dict:
    """Rolling answer-quality aggregates for the SLO page."""
    with read_only_connection() as conn:
        summary = conn.execute(
            QUALITY_SUMMARY, {"hours": hours, "low_ceiling": LOW_FAITHFULNESS_CEILING}
        ).fetchone()
        recent = conn.execute(RECENT_EVALUATIONS, {"hours": hours, "limit": recent_limit}).fetchall()

    scored = int(summary["scored"] or 0) if summary else 0

    metrics = [
        _metric_row("faithfulness", summary["faithfulness_mean"], summary["faithfulness_p50"], scored),
        _metric_row("answer_accuracy", summary["answer_accuracy_mean"], summary["answer_accuracy_p50"], scored),
        _metric_row(
            "context_precision", summary["context_precision_mean"], summary["context_precision_p50"], scored
        ),
        _metric_row("context_recall", summary["context_recall_mean"], summary["context_recall_p50"], scored),
    ] if summary else []

    low = int(summary["low_faithfulness"] or 0) if summary else 0

    return {
        "window_hours": hours,
        "scored": scored,
        "skipped": int(summary["skipped"] or 0) if summary else 0,
        "failed": int(summary["failed"] or 0) if summary else 0,
        "metrics": metrics,
        "low_faithfulness_count": low,
        "low_faithfulness_rate": round(low / scored, 4) if scored else None,
        "low_faithfulness_ceiling": LOW_FAITHFULNESS_CEILING,
        "recall_basis": "generated_answer",
        "recent": [
            {
                "request_id": row["request_id"],
                "evidence_path": row["evidence_path"],
                "status": row["status"],
                "faithfulness": _clean(row["faithfulness"]),
                "answer_accuracy": _clean(row["answer_accuracy"]),
                "context_precision": _clean(row["context_precision"]),
                "context_recall": _clean(row["context_recall"]),
                "created_at": row["created_at"].isoformat(),
            }
            for row in recent
        ],
    }
