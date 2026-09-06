import argparse
import json
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.database import close_pool, open_pool
from configs.settings import settings
from src.auth.rbac import access_scopes_for
from src.graph.builder import get_compiled_graph
from src.graph.state import initial_state
from src.schemas.enums import Role, TerminalOutcome

GOLDEN_SET = Path(__file__).parent / "golden_set.json"

SLO = {
    "task_success_rate": 0.90,
    "sql_correctness": 0.95,
    "high_risk_misclassification": 0.05,
    "pii_leakage": 0.0,
    "confidence_present": 1.0,
    "citation_coverage": 0.95,
    "escalation_precision": 0.85,
    "p95_standard_seconds": 4.0,
    "p95_hybrid_seconds": 6.0,
    "p95_agentic_seconds": 10.0,
    "p95_high_risk_seconds": 12.0,
}

DEADLINE_FOR_PATH = {
    "rag": settings.deadline_seconds_standard,
    "nl2sql": settings.deadline_seconds_standard,
    "hybrid": settings.deadline_seconds_hybrid,
    "agentic": settings.deadline_seconds_agentic,
    "high_risk_panel": settings.deadline_seconds_high_risk,
}


@dataclass
class CaseResult:
    case_id: str
    query: str
    elapsed_seconds: float
    outcome: str
    risk_level: str | None = None
    evidence_path: str | None = None
    confidence: float | None = None
    citations: int = 0
    template_id: str | None = None
    routing_correct: bool = False
    risk_correct: bool = False
    escalation_correct: bool = False
    error: str = ""


@dataclass
class Report:
    total: int = 0
    results: list[CaseResult] = field(default_factory=list)

    def by_expected_path(self, path: str) -> list[CaseResult]:
        return [r for r in self.results if r.evidence_path == path]


def run_case(case: dict) -> CaseResult:
    role = Role(case.get("role", "compliance_officer"))
    expected_path = case.get("expected_path")

    deadline_budget = DEADLINE_FOR_PATH.get(expected_path or "rag", settings.deadline_seconds_standard)

    thread_id = str(uuid.uuid4())

    state = initial_state(
        request_id=str(uuid.uuid4()),
        thread_id=thread_id,
        user_id=f"eval-{role.value}",
        role=role.value,
        access_scopes=access_scopes_for(role),
        raw_query=case["query"],
        deadline_ts=time.monotonic() + deadline_budget,
        token_budget=settings.default_token_budget,
    )

    state["departments"] = []

    started = time.monotonic()

    try:
        final = get_compiled_graph().invoke(
            state, config={"configurable": {"thread_id": thread_id}, "recursion_limit": 40}
        )
    except Exception as exc:
        return CaseResult(
            case_id=case["id"],
            query=case["query"],
            elapsed_seconds=round(time.monotonic() - started, 3),
            outcome="error",
            error=str(exc)[:200],
        )

    elapsed = round(time.monotonic() - started, 3)

    outcome = final.get("terminal_outcome") or "unknown"
    risk = final.get("risk")
    confidence = final.get("confidence")
    evidence = final.get("sql_evidence")

    result = CaseResult(
        case_id=case["id"],
        query=case["query"],
        elapsed_seconds=elapsed,
        outcome=outcome,
        risk_level=risk.final_level.value if risk else None,
        evidence_path=final.get("evidence_path"),
        confidence=confidence.final_score if confidence else None,
        citations=len(final.get("retrieved_chunks", [])),
        template_id=evidence.template_id if evidence else None,
    )

    if expected_path:
        result.routing_correct = result.evidence_path == expected_path

    if case.get("expected_risk"):
        result.risk_correct = result.risk_level == case["expected_risk"]

    should_escalate = case.get("should_escalate")

    if should_escalate is not None:
        escalated = outcome == TerminalOutcome.ESCALATED.value
        result.escalation_correct = escalated == should_escalate

    if case.get("expected_outcome"):
        expected = case["expected_outcome"]

        if expected == "refused_or_scoped":
            result.escalation_correct = outcome in (
                TerminalOutcome.REFUSED.value,
                TerminalOutcome.ANSWERED.value,
            )
        else:
            result.escalation_correct = outcome == expected

    return result


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    index = min(len(ordered) - 1, round(p * (len(ordered) - 1)))

    return ordered[index]


def summarise(report: Report) -> dict:
    results = report.results

    routed = [r for r in results if r.evidence_path]
    risk_scored = [r for r in results if r.risk_level]

    high_risk_cases = [r for r in results if r.case_id.startswith("hr-")]
    misclassified = [r for r in high_risk_cases if r.risk_level != "High"]

    escalation_cases = [r for r in results if r.case_id.startswith(("hr-", "adv-"))]
    escalation_correct = [r for r in escalation_cases if r.escalation_correct]

    answered = [r for r in results if r.outcome == TerminalOutcome.ANSWERED.value]
    with_confidence = [r for r in results if r.confidence is not None]

    sql_cases = [r for r in results if r.case_id.startswith("rl-")]
    sql_ok = [r for r in sql_cases if r.template_id]

    errors = [r for r in results if r.outcome == "error"]

    latency_by_path = {}

    for path in ("rag", "nl2sql", "hybrid", "agentic", "high_risk_panel"):
        group = [r.elapsed_seconds for r in results if r.evidence_path == path]

        if group:
            latency_by_path[path] = {
                "count": len(group),
                "p50": round(statistics.median(group), 3),
                "p95": round(percentile(group, 0.95), 3),
            }

    return {
        "cases_run": len(results),
        "errors": len(errors),
        "task_success_rate": round((len(results) - len(errors)) / len(results), 4) if results else 0.0,
        "routing_accuracy": round(
            sum(1 for r in routed if r.routing_correct) / len(routed), 4
        ) if routed else 0.0,
        "risk_accuracy": round(
            sum(1 for r in risk_scored if r.risk_correct) / len(risk_scored), 4
        ) if risk_scored else 0.0,
        "high_risk_misclassification": round(
            len(misclassified) / len(high_risk_cases), 4
        ) if high_risk_cases else 0.0,
        "escalation_precision": round(
            len(escalation_correct) / len(escalation_cases), 4
        ) if escalation_cases else 0.0,
        "confidence_present": round(len(with_confidence) / len(results), 4) if results else 0.0,
        "sql_template_selected": round(len(sql_ok) / len(sql_cases), 4) if sql_cases else 0.0,
        "answered_count": len(answered),
        "latency_by_path": latency_by_path,
    }


def check_slo(summary: dict) -> list[str]:
    breaches = []

    if summary["task_success_rate"] < SLO["task_success_rate"]:
        breaches.append(f"task success {summary['task_success_rate']} < {SLO['task_success_rate']}")

    if summary["high_risk_misclassification"] > SLO["high_risk_misclassification"]:
        breaches.append(
            f"high-risk misclassification {summary['high_risk_misclassification']} > {SLO['high_risk_misclassification']}"
        )

    if summary["escalation_precision"] < SLO["escalation_precision"]:
        breaches.append(
            f"escalation precision {summary['escalation_precision']} < {SLO['escalation_precision']}"
        )

    if summary["confidence_present"] < SLO["confidence_present"]:
        breaches.append(f"confidence present {summary['confidence_present']} < 1.0")

    for path, budget_key in (
        ("rag", "p95_standard_seconds"),
        ("nl2sql", "p95_standard_seconds"),
        ("hybrid", "p95_hybrid_seconds"),
        ("agentic", "p95_agentic_seconds"),
        ("high_risk_panel", "p95_high_risk_seconds"),
    ):
        stats = summary["latency_by_path"].get(path)

        if stats and stats["p95"] > SLO[budget_key]:
            breaches.append(f"{path} p95 {stats['p95']}s > {SLO[budget_key]}s")

    return breaches


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the golden set against the compiled graph")
    parser.add_argument("--filter", default="", help="only run case ids starting with this prefix")
    parser.add_argument("--out", default="evals/last_run.json")
    parser.add_argument("--ci", action="store_true", help="exit non-zero when an SLO is breached")
    args = parser.parse_args()

    payload = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    cases = payload["cases"]

    if args.filter:
        cases = [c for c in cases if c["id"].startswith(args.filter)]

    open_pool()
    report = Report(total=len(cases))

    try:
        for case in cases:
            result = run_case(case)
            report.results.append(result)
            print(f"{result.case_id:8s} {result.outcome:24s} {result.elapsed_seconds:6.2f}s {result.error}")
    finally:
        close_pool()

    summary = summarise(report)
    breaches = check_slo(summary)

    output = {
        "summary": summary,
        "slo_breaches": breaches,
        "results": [asdict(r) for r in report.results],
    }

    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\n" + json.dumps(summary, indent=2))

    if breaches:
        print("\nSLO breaches:")

        for breach in breaches:
            print(f"  - {breach}")

    if args.ci and breaches:
        sys.exit(1)


if __name__ == "__main__":
    main()
