"""Run the by-role SLO dataset against the live /ask API and fill in 'SLO Results'.

Reads every test row from ``evals/test_dataset_50_by_role_slo.xlsx`` (sheet
'Test Dataset'), POSTs each question to ``{base-url}/ask`` with ``use_cache``
off and a fresh thread, waits for the background RAGAS evaluation of answered
rows, and writes what happened into the 'SLO Results' sheet:

    J  Actual HTTP          M  Faithfulness
    K  Actual Outcome       N  Context Precision
    L  Latency (ms)         O  Context Recall
    V  Run Notes            (accuracy verdict, evidence path, token-role mismatches)

The PASS/FAIL columns and the 'SLO Summary' sheet are formulas and fill
themselves when the workbook is opened in Excel or Numbers.

Auth: the API only trusts Auth0 tokens. Log in to the frontend as each role,
copy the Authorization bearer from any /ask call (browser dev tools), and
export it. Per-role tokens win over the shared fallback:

    export EVAL_TOKEN_STORE_ASSOCIATE="eyJ..."
    export EVAL_TOKEN_STORE_MANAGER="eyJ..."
    export EVAL_TOKEN_COMPLIANCE_OFFICER="eyJ..."
    export EVAL_TOKEN_LEGAL_REVIEWER="eyJ..."
    export EVAL_TOKEN_ADMIN="eyJ..."
    export EVAL_TOKEN="eyJ..."           # fallback for any role above

The Azure runtime credentials must already be registered for each token's user
(log in once through the frontend so /auth/azure has run), or /ask will refuse.

Usage:
    uv run python evals/run_slo_dataset.py                        # localhost:8000
    uv run python evals/run_slo_dataset.py --base-url https://rpids-api.onrender.com
    uv run python evals/run_slo_dataset.py --filter nm-           # only the traps
    uv run python evals/run_slo_dataset.py --limit 3              # smoke run
    uv run python evals/run_slo_dataset.py --in-place             # write the master file
"""

import argparse
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
from openpyxl import load_workbook

EVALS_DIR = Path(__file__).parent
WORKBOOK = EVALS_DIR / "test_dataset_50_by_role_slo.xlsx"

ASK_TIMEOUT_SECONDS = 180.0
EVAL_POLL_INTERVAL_SECONDS = 5.0


@dataclass
class Case:
    row: int  # 1-based row in 'SLO Results' (same order as 'Test Dataset')
    test_id: str
    question: str
    role: str
    expected_outcome: str
    ragas_scored: bool


def token_for(role: str) -> str | None:
    return os.environ.get(f"EVAL_TOKEN_{role.upper()}") or os.environ.get("EVAL_TOKEN")


def load_cases(path: Path) -> list[Case]:
    wb = load_workbook(path, read_only=True)
    td = wb["Test Dataset"]
    cases = []

    for row in td.iter_rows(min_row=2, values_only=False):
        test_id = row[1].value  # column B

        if not test_id:  # a role-group separator row
            continue

        cases.append(
            Case(
                row=row[0].row,
                test_id=str(test_id),
                question=str(row[3].value),
                role=str(row[4].value),
                expected_outcome=str(row[8].value),
                ragas_scored=str(row[21].value).strip().lower() == "yes",
            )
        )

    wb.close()

    return cases


def ask(client: httpx.Client, base_url: str, token: str, question: str) -> tuple[int, dict, float]:
    payload = {"query": question, "thread_id": str(uuid.uuid4()), "use_cache": False}
    started = time.perf_counter()
    response = client.post(
        f"{base_url}/ask",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=ASK_TIMEOUT_SECONDS,
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    try:
        body = response.json()
    except ValueError:
        body = {}

    return response.status_code, body, latency_ms


def wait_for_evaluation(
    client: httpx.Client, base_url: str, token: str, request_id: str, timeout_seconds: float
) -> dict | None:
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        response = client.get(
            f"{base_url}/requests/{request_id}/evaluation",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        )

        if response.status_code == 200:
            body = response.json()

            if body.get("status") != "pending":
                return body

        time.sleep(EVAL_POLL_INTERVAL_SECONDS)

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--workbook", default=str(WORKBOOK))
    parser.add_argument("--filter", default="", help="only run test ids starting with this prefix (e.g. nm-)")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many cases (0 = all)")
    parser.add_argument("--eval-timeout", type=float, default=150.0, help="seconds to wait for the RAGAS scores")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds to sleep between requests")
    parser.add_argument("--in-place", action="store_true", help="write results into the master workbook itself")
    args = parser.parse_args()

    source = Path(args.workbook)
    cases = load_cases(source)

    if args.filter:
        cases = [c for c in cases if c.test_id.startswith(args.filter)]

    if args.limit:
        cases = cases[: args.limit]

    if not cases:
        print("no cases matched")

        return 1

    missing = sorted({c.role for c in cases if token_for(c.role) is None})

    if missing:
        print(f"no token for role(s): {', '.join(missing)}")
        print("export EVAL_TOKEN_<ROLE> or EVAL_TOKEN (see the module docstring), then rerun")

        return 1

    wb = load_workbook(source)
    results = wb["SLO Results"]
    # test id -> row in 'SLO Results'
    row_of = {
        str(results.cell(r, 2).value): r
        for r in range(2, results.max_row + 1)
        if results.cell(r, 2).value
    }

    base_url = args.base_url.rstrip("/")
    ran = failed = 0

    with httpx.Client() as client:
        for index, case in enumerate(cases, 1):
            token = token_for(case.role)
            notes: list[str] = []

            print(f"[{index}/{len(cases)}] {case.test_id} ({case.role}): {case.question[:70]}", flush=True)

            try:
                status_code, body, latency_ms = ask(client, base_url, token, case.question)
            except Exception as exc:
                print(f"    request failed: {exc}")
                status_code, body, latency_ms = 0, {}, 0.0
                notes.append(f"request error: {str(exc)[:120]}")

            outcome = body.get("status") or ""
            request_id = body.get("request_id") or ""
            row = row_of.get(case.test_id)

            if row is None:
                print(f"    test id {case.test_id} not found on 'SLO Results', skipping write")

                continue

            results.cell(row, 10).value = status_code or None  # J Actual HTTP
            results.cell(row, 11).value = outcome or None      # K Actual Outcome
            results.cell(row, 12).value = latency_ms or None   # L Latency (ms)

            if body.get("evidence_path"):
                notes.append(f"path={body['evidence_path']}")

            if case.ragas_scored and outcome == "answered" and request_id:
                evaluation = wait_for_evaluation(client, base_url, token, request_id, args.eval_timeout)

                if evaluation is None:
                    notes.append("ragas: timed out waiting for scores")
                elif evaluation.get("status") == "done":
                    results.cell(row, 13).value = evaluation.get("faithfulness")        # M
                    results.cell(row, 14).value = evaluation.get("context_precision")   # N
                    results.cell(row, 15).value = evaluation.get("context_recall")      # O
                    accuracy = evaluation.get("answer_accuracy")
                    notes.append(f"accuracy={accuracy if accuracy is not None else 'not scored'}")
                else:
                    notes.append(f"ragas: {evaluation.get('status')} ({evaluation.get('skipped_reason') or evaluation.get('error') or ''})".strip())

            ok = outcome == case.expected_outcome
            ran += 1
            failed += 0 if ok else 1
            print(f"    http={status_code} outcome={outcome or '-'} ({'OK' if ok else 'MISMATCH, expected ' + case.expected_outcome}) {latency_ms:.0f} ms")

            if notes:
                results.cell(row, 22).value = " · ".join(notes)  # V Run Notes

            time.sleep(args.delay)

    if args.in_place:
        out = source
    else:
        out = source.with_name(f"{source.stem}_run_{datetime.now():%Y%m%d_%H%M}.xlsx")

    wb.save(out)
    print(f"\n{ran} case(s) run, {failed} outcome mismatch(es)")
    print(f"results written to {out}")
    print("open it in Excel or Numbers - the PASS/FAIL columns and 'SLO Summary' recalculate on open")

    return 0


if __name__ == "__main__":
    sys.exit(main())
