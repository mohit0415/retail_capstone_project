# Capstone alignment — where the backend stands against the brief

Assessed on 2026-09-07 against `README 1.md` (the capstone brief: *Retail Policy Intelligence &
Decision Support System, SLO-Bound Autonomous Agentic AI System*), after the logging, caching
and model-routing work of README §18. Status legend: **Met** — implemented and covered by tests;
**Partly** — implemented, but evidence or a piece is missing; **Gap** — not in the repository.

## 1. Core requirements

| # | requirement (brief) | status | where / what is missing |
|---|---|---|---|
| 1 | Understand intent; classify risk Low/Medium/High; detect structured lookup vs interpretation; multi-turn context; RBAC | **Met** | `intent_classification`, `risk_assessment` (L1 lexical → L2 classifier → L3 DB probes, fused), `planner` + `path_policy`, `conversation.py` (window, rewrite, summary), `src/auth/rbac.py` (5 roles → document/table scopes) |
| 2 | Route dynamically to RAG / SQL / Hybrid / Multi-agent | **Met** | `route_evidence_path` + `path_policy.py` decision table with capability gates and budget degradation; High risk → panel unconditionally |
| 3 | Five agents in Plan–Reason–Act with reflection | **Met** | see `docs/AGENT_WORKFLOW.md`; `planner → evidence → compliance_validation → reflection → planner`, bounded by `MAX_REFLECTION_RETRIES` and the budget |
| 4 | Every response: citations, structured validation output, confidence score, risk classification, uncertainty disclosure | **Met** | `AnswerResponse`: `citations`, `sql_evidence`, `confidence`, `risk_level`, `uncertainty_note`, full `trace` (validation, confidence breakdown, routing, usage) |
| 5 | Escalate on low confidence / High risk / conflict / explicit legal request, with full context transfer | **Met** | `after_confidence`, `escalation_manager`; `build_context_package` carries history, retrieved documents, SQL evidence, validation, panel verdict, plan, reasoning trace; reviewer resumes the paused graph |

## 2. Success criteria (SLOs)

| SLO (brief) | status | evidence and caveat |
|---|---|---|
| Task Success Rate ≥ 90 % | **Partly** | `evals/run_eval.py` computes `task_success_rate` over the 53-case golden set. It has to be *run* against a live stack and the report kept — no run output is in the repo yet. |
| P95 latency ≤ defined threshold (e.g. 3–6 s) | **Partly** | Targets are defined per stage and per path (`SLO_T1..T4_P95_MS`, `SLO_PATH_TARGET_RATIO`) and measured in `request_latency` → `GET /metrics/slo`. A cold multi-agent run (7+ model calls) is engineered for 30–90 s deadlines, not 3–6 s; cache hits now answer in milliseconds and routing trims the generation calls. State the per-path targets as *the* SLO in the report and show cache-hit latency separately. Note the eval's `p95_*_seconds` (4/6/10/12 s) are tighter than the runtime targets — reconcile them. |
| Zero PII leakage | **Partly** | Presidio + regex redaction on input, Guardrails-AI DetectPII + Presidio on output, PII entities logged and audited. The eval declares `pii_leakage: 0.0` but does not probe it — add cases that plant PII in the question and assert it never appears in the answer. |
| Structured SQL correctness ≥ 95 % | **Partly** | Vetted templates, shape + scope guards, disclosure caveats, 40+ SQL tests. The eval only measures *template selected* (`sql_template_selected`), not result correctness — add expected row counts / ids to the `rl-*` cases. |
| High-risk misclassification < 5 % | **Met (measured)** | `high_risk_misclassification` in the eval over the 10 `hr-*` cases. |
| Cost per query within budget | **Met (new)** | `src/llm_routing/ledger.py` prices every billed call; `trace.llm_usage` per request, `GET /metrics/optimization` process-wide, `COST_BUDGET_USD_PER_REQUEST` soft ceiling with `requests_over_budget`. llama-index calls (agentic path, generated SQL) are outside the ledger — documented. |

## 3. Technical scope (six layers)

| layer | status | notes |
|---|---|---|
| API (FastAPI, auth, validation) | **Met** | JWT, RBAC, pydantic validation, rate limiting, idempotency, request-id middleware |
| Orchestration (LangGraph) | **Met** | 18-node graph, Postgres checkpointer, interrupt for review |
| Retrieval & knowledge (RAG + SQL) | **Met** | pgvector + BM25 fusion + FlashRank, version filter; vetted + generated SQL |
| External tools (MCP) | **Met when configured** | `src/tools/mcp_tools.py` behind `MCP_SERVER_URL`; optional LiteLLM gateway |
| Evaluation & observability (Langfuse, metrics, tracing) | **Met** | Langfuse traces + prompt registry, request-id-stamped logs, `/metrics/slo`, `/metrics/optimization`, golden-set eval |
| Human-in-the-loop (escalation + audit) | **Met** | escalation queue, e-mail for High risk, append-only audit with circuit breaker, review resume |

## 4. Dataset guidance

**Met.** Seven public / synthetic policy documents in `data/policies` (GDPR excerpts, ISO 27001
summary, five internal policies) and the four required tables `vendors`, `audit_logs`,
`retention_records`, `compliance_reviews` with synthetic seed data (`data/sql`). No proprietary
material.

## 5. Mandatory high-risk scenarios

The code paths for all eight are present (table in `docs/AGENT_WORKFLOW.md`). The golden set
covers deletion under legal hold, breach notification, termination over a Critical finding,
kickbacks, escalated findings, overdue remediation, whistleblowing and lawful basis. **Add
explicit cases** for the four scenarios not yet named in `evals/golden_set.json`: cross-border
transfer to a restricted jurisdiction, approval override for a Critical-risk vendor,
conflicting clauses across departments, and a high-severity issue marked Closed without
evidence — so the < 5 % misclassification figure is measured on the brief's own list.

## 6. Deliverables

| # | deliverable | status | where |
|---|---|---|---|
| 1 | Architecture diagram | **Met (new)** | `docs/ARCHITECTURE.md` (Mermaid, six layers) |
| 2 | Agent workflow diagram | **Met (new)** | `docs/AGENT_WORKFLOW.md` (Mermaid, every node and edge) |
| 3 | RAG + SQL integration | **Met** | `src/retrieval`, `src/sqlpath`, `hybrid_path` |
| 4 | Multi-agent orchestration | **Met** | `src/graph`, `src/nodes/multi_agent_panel.py` |
| 5 | SLO definition & evaluation report | **Partly** | SLOs defined in `.env.example` / `configs/settings.py` and README §3; run `python evals/run_eval.py` against the live stack and commit the output (plus a `GET /metrics/slo` snapshot) as the report |
| 6 | Observability dashboard evidence | **Partly** | Langfuse is wired (`TRACING_ENABLED=true` + keys); capture a Langfuse trace screenshot and a `GET /metrics/optimization` snapshot after the eval run |
| 7 | Escalation workflow implementation | **Met** | `escalation_manager`, `/review/*`, resume through the output guardrail |
| 8 | 4–6 minute live demo | **Pending** | suggested script in §8 below |
| 9 | Deployment & runbook documentation | **Partly** | `docs/API_RUN_BOOK.pdf`, `docs/Swagger_Runbook_*.pdf`, README §12; no `Dockerfile` / `docker-compose.yml` — add one that brings up Postgres+pgvector and the API |

## 7. What changed in this pass (README §18)

- Request-correlated structured logging across every layer; access log; startup summary; the
  stray `configs/logger.py` neutralised.
- Response cache (exact + semantic, access-scoped, certified answers only), retrieval cache,
  LLM completion cache; invalidation on corpus change; `/metrics/optimization`,
  `/cache/invalidate`; cache evidence in the audit log and SLO history.
- Heuristic model routing (Sprint-9 pattern) for the four expensive calls, budget-pressure
  downgrade, cost ledger with real token usage, optional LiteLLM gateway.
- Two pre-existing test failures fixed (`/requests/{id}` `queued_at` KeyError; stale test
  fixture); ruff clean; 543 tests passing.

## 8. Suggested demo script (4–6 minutes)

1. `GET /health` — layers up, routing strategy, caches enabled.
2. `POST /ask` as `store_manager`: "What is the retention period for customer invoices?" —
   answer with citation, `trace.model_routing` shows `rag_generate=small`, `llm_usage.usd`.
3. Repeat the question as another `store_manager` — `cache.kind = exact`, `timings.total_ms`
   in milliseconds; rephrase it — `cache.kind = semantic`.
4. `POST /ask` as `compliance_officer`: "Can we dispose of records under legal hold because their
   retention period has elapsed?" — risk High → panel → 202 with `poll_url` and
   `escalation_reference`; show the log lines for the panel stages.
5. `GET /review/queue` → `POST /review/{id}` accept → `GET /requests/{id}` answered.
6. `GET /metrics/slo` and `GET /metrics/optimization` — P95 per path, hit rate, small-tier
   share, spend per request; the Langfuse trace for step 4.

## 9. Verdict

The system meets the brief's functional requirements and architecture, and it now measures
cost as well as latency and quality. What separates it from a finished capstone is evidence,
not code: run the evaluation and keep its output, capture the observability screenshots, extend
the golden set to the brief's own high-risk list (and add PII / SQL-correctness probes), and
add a container recipe to the runbook.
