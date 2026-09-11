# Retail Policy Intelligence & Decision Support System — backend

An agentic compliance question-answering service over retail policy documents and operational
records. FastAPI + LangGraph orchestration, LlamaIndex knowledge layer, Azure OpenAI, PostgreSQL with
pgvector.

This file carries the conceptual explanation of the code. The code itself has no comments, so
anything that needs explaining is explained here.

---

## 1. The shape of a request

A question enters at `POST /ask` and walks one graph. The graph has five kinds of node and every one
of them can end the request:

```
                    input_guardrail
                          |
                    query_rewrite  (uses prior turns + thread summary)
                          |
                  intent_classification
                          |
                   entity_resolution ------- ambiguous --> clarification
                          |
                    risk_assessment  (L1 lexical, L2 classifier, L3 SQL probe)
                          |
                       planner  (route: rag | nl2sql | hybrid | high_risk_panel)
                          |
        rag_path --> nl2sql_path --> multi_agent_panel     (only the stages the route needs,
                          |                                  always in this order)
                          |---- evidence does not answer it --> no_answer ("I don't know")
                          |
                 compliance_validation --- defects --> reflection --> planner
                          |
                  confidence_scoring
                          |
                  output_guardrail  <---- human review resumes here
                          |
                  answered   or   escalation_manager --> review queue
```

Three things about this shape are worth stating because they are easy to get wrong.

**Risk does not choose the path.** Only `High` forces a path, and the path it forces is the
multi-agent panel. `Low` and `Medium` are treated identically by the router. Which of rag / nl2sql /
hybrid runs is decided by the **intent**, not by the planner's free choice: the intent fixes a small
set of admissible paths and the planner may only pick inside it. Section 15 is the decision table and
the reasoning behind it. (The v4 workflow has no agentic route; no intent admits one.)

**The evidence stages have one order.** RAG first, then NL2SQL, then the multi-agent panel, then
compliance validation. A route runs only the stages it needs — `rag` is `rag_path`, `nl2sql` is
`nl2sql_path`, `hybrid` is `rag_path → nl2sql_path` (the records stage writes the reconciled
answer), High risk is `rag_path → nl2sql_path → multi_agent_panel` — but never in another order.
`docs/AGENT_WORKFLOW.md` has the full diagram.

**Every terminal state is a real state.** `answered`, `escalated`, `refused` and
`clarification_required` are the only four ways out. A request that cannot be certified escalates; it
never returns a hedge. A request whose evidence simply does not contain the answer is *not*
escalated: it is answered with an honest "I don't know" (`no_answer`, `not_found: true`), because a
reviewer cannot certify an answer nobody can source.

**Escalation is a node, not a return code.** Anything that decides the answer cannot be released
routes into `escalation_manager`, which writes the review queue row. That includes the output
guardrail, which previously ended the graph directly and produced a `202` whose `poll_url` pointed at
a review record that had never been created.

---

## 2. Budgets, deadlines and why they were the root of everything

Every node runs under a wall-clock deadline and a token budget carried in state. `traced_node` checks
the budget **before** the node body runs, and there are three tiers:

- **Optional tier** (`reranker`, `thread_summary`, `confidence_calibration`): skipped under pressure.
  Skipping sets `degraded` and costs 0.10 confidence, and the skip is named in
  `skipped_optional_nodes`.
- **Required tier** (everything that gathers or judges evidence): not run under pressure. The node
  records a `budget_stops` entry and sets `escalation_reason`, and the router sends the request to
  human review rather than answering from a half-finished pipeline.
- **Release tier** (`output_guardrail`, `escalation_manager`, `safe_refusal`, `clarification`):
  never budget-checked. By the time a request reaches these the budget is usually spent, and a
  budget check here would prevent the system from escalating — the exact failure it is meant to
  produce.

### The deadline follows the route, it does not guess at it

`/ask` stamps the standard deadline on every request. It cannot do better than that: at t0 nothing
has classified the question yet, so any richer stamp is a guess at the query text. The budget is then
widened by the two nodes that actually settle the route:

- `risk_assessment` widens to `DEADLINE_SECONDS_HIGH_RISK` the moment fusion returns High, because a
  High verdict makes the three-agent panel mandatory.
- `planner` widens to the deadline belonging to whichever path it picked — hybrid, agentic or the
  panel.

Both call `widen_for_path`, which only ever extends. Two properties matter and both are tested. The
new deadline is computed from `started_ts`, not from the moment of widening, so a path that widens
late does not restart the clock and get more total time than its budget allows. And a cheaper
re-plan after reflection can never claw back time already granted — a request that reached High risk
keeps the wide budget for the rest of its life.

An earlier version stamped the deadline from lexical markers in the query. It was wrong in both
directions. A High-risk question phrased without a trigger word ran the mandatory panel — five model
calls plus retrieval plus a SQL read — on the 30-second standard budget and was stopped by the guard,
producing an escalation caused by a keyword table rather than by anything about the answer. And a
Low-risk question that happened to contain "penalty" was handed 90 seconds it had no use for, so it
could run four times over the standard objective without anything reporting a breach.

**The failure this caused.** A `.env` shipping `DEADLINE_SECONDS_STANDARD=4.0` gave a request four
seconds. A single `/ask` makes roughly seven Azure OpenAI calls plus an embedding call plus a
FlashRank load, so the deadline was blown before the answer was even drafted — on every request. The
visible symptom was a `202` on every question regardless of risk, but three quieter things were also
happening:

- FlashRank never ran, because the reranker is optional-tier and an exhausted deadline degrades it.
  That set `degraded` (−0.10 confidence) on every answer and left `rerank_score` unset, so the
  confidence scorer fell back to raw cosine or RRF scores.
- The agentic and hybrid paths never executed, because `route_evidence_path` degrades agentic to
  hybrid below `AGENTIC_MIN_SECONDS` and hybrid to rag below 1.5s.
- The reflection repair loop never ran once, because it needs a second of headroom to start.

`configs/settings.py` enforces a floor (`MINIMUM_DEADLINES`, `MINIMUM_TOKEN_BUDGET`) and logs a
warning naming every field it had to raise. **Treat that warning on startup as a bug in your `.env`,
not as a safety net doing its job.** The floor is 30 / 45 / 75 / 90 seconds and 16000 tokens; the
shipped `.env` and `.env.example` configure 30 / 45 / 75 / 90 and 24000.

The token budget is 24000 rather than the 16000 floor because the floor was sized for the RAG path.
Adding up `NODE_TOKEN_ESTIMATE`, an intake plus a panel pass plus its repair costs roughly 15000, and
the reflection loop then needs another re-plan, another evidence pass and another validation on top.
16000 covers a RAG request with one repair; it leaves the panel with nothing to reflect with.

---

## 3. t0–t4 timing and the SLO report

Each node stamps a mark carrying the node name, its stage, and milliseconds elapsed since `t0`.
`t0` is the moment `/ask` accepted the request, recorded as `started_ts` in the initial state, so
every figure is measured against one clock rather than stitched together from node durations.

| point | meaning |
|-------|---------|
| t0 | `/ask` accepted the request |
| t1 | intake complete: guardrail, rewrite, intent, entities, risk |
| t2 | evidence plan ready |
| t3 | evidence gathered and the answer drafted |
| t4 | validated, scored, and released or escalated |

Stage marks are cumulative from t0, so they read as a timeline rather than a set of durations. If a
request escalates before reaching a stage, that stage carries forward the last value reached instead
of leaving a null hole — otherwise percentiles would be computed over a shifting population and
`t3_p95` would silently exclude every request that failed early, which is the population you most
want to see.

One row per request lands in `request_latency`. `GET /metrics/slo` computes P50/P95/P99 per stage
using Postgres `percentile_cont`, compares each P95 against its configured target
(`SLO_T1_P95_MS` … `SLO_T4_P95_MS`), and returns `breached_stages` naming the ones that missed.
The report also breaks the window down by outcome and by evidence path, because a P95 regression is
almost always one path getting slower rather than everything getting slower.

### Why each path is scored against its own target

The paths are not given the same budget, so scoring them against one number makes the report lie.
A high-risk request is allowed 90 seconds by design; judged against a single 28-second objective it
reads as a breach every time, and a report that cries breach on correct behaviour is a report nobody
reads.

Each path therefore carries `target_p95_ms` of its own, computed as
`path deadline × SLO_PATH_TARGET_RATIO` (0.85). The ratio is what makes the target an objective
rather than a restatement of the hard stop: the deadline is where the guard stops the request, and
the target says a path should normally finish inside 85% of it. A path whose P95 sits at its own
deadline is not healthy — it is one slow model call away from being stopped — and `breached_paths`
names it before that starts happening.

`total.target_p95_ms` is the sample-weighted mean of the per-path targets actually seen in the
window, so a window holding a lot of high-risk traffic is judged against the budget that traffic was
really given. `total_target_basis` in the response says so, because an unexplained moving target
looks like a bug.

The response to `/ask` also carries its own `timings` block, so a single call shows where its time
went without querying the metrics endpoint.

---

## 4. Vetted SQL templates

**No SQL is generated at runtime.** The previous design handed the question to LlamaIndex's
`NLSQLTableQueryEngine`, let it write SQL, and validated the result. Validation after generation
cannot make a generated query *correct* — it can only make it non-destructive. A query that quietly
reads `vendors.risk_category` when the question was about `audit_logs.issue_severity` passes every
safety check and returns a confident wrong answer.

The SQL path now works like this:

1. `src/sqlpath/templates.py` holds a registry of 17 reviewed queries. Each declares its
   `template_id`, what it answers, the tables it touches, and its typed parameters.
2. `src/sqlpath/selector.py` shows the model only the templates the caller's role may run, and asks
   it to pick one `template_id` and supply that template's parameters — or to say the question is not
   answerable from the catalogue. It cannot return SQL, because SQL is not a field in the schema it
   fills.
3. `bind_parameters` coerces and validates every value. Integers must parse. Enum parameters must
   match one of the stored values, case-insensitively, and are rewritten to the exact stored spelling
   — this matters because the value sets are stored with capitals and spaces (`Non-Compliant`, not
   `non_compliant`), and a near-miss returns zero rows, which reads as "no problems found".
4. A `vendor_id` is taken from entity resolution, not from the model. If the question names a vendor
   that was never resolved, the selector is told to use `vendor_search_by_name` instead of inventing
   an id.
5. The static template text still passes `validate_generated_sql` and `assert_tables_in_scope` before
   execution. These can no longer fail in practice, and that is the point: they are a tripwire that
   fires if someone later adds a template with a clock read or an out-of-scope join.
6. Execution goes through `read_only_connection` with parameters bound by psycopg. Values never
   reach the SQL text.

Refusal is a first-class outcome. If nothing in the catalogue answers the question, the path returns
"no vetted query answers this" and the request escalates. That is strictly better than a generated
query that answers a slightly different question.

**Role scoping happens twice.** `templates_visible_to` filters the catalogue to templates whose
tables are a subset of the role's grant, so a `store_associate` (no table grants) sees zero queries
and a `store_manager` sees 9 of 17. After execution, `_scope_rows` drops rows outside the caller's
department or risk-category scope and reports the count, which the narration must disclose as a
caveat — a filtered count is a floor, not a total.

**Nothing reads the wall clock.** Every date comparison binds `as_of` from `settings.as_of_date`.
The seed data is dated 2025, so a `CURRENT_DATE` anywhere would mark the whole corpus overdue and
fire every risk probe. Beyond that, a compliance figure without an as-of date is not reproducible and
cannot be audited.

---

## 5. Multi-turn conversation

`query_rewrite` has always been able to use prior turns; it simply never received any, because every
request started with `conversation_history=[]` and nothing ever wrote a turn back.

`src/core/conversation.py` now persists turns in `conversation_turns` keyed by `thread_id`. `/ask`
loads the last 10 turns plus the thread summary when the caller supplies a `thread_id`, and writes
the user and assistant turns back after a successful answer. The write runs as a FastAPI background
task so summarisation latency lands after the response, not inside it.

Once a thread passes 8 turns, `refresh_summary` compresses the older history into a rolling summary
held in `conversation_threads`. The summary keeps the things a later turn needs to resolve a pronoun
— vendors, departments, clauses, record ids and what was concluded about each — and drops everything
else. A window plus a summary keeps the rewrite prompt bounded no matter how long the thread runs.

Answers released by a human reviewer are written back to the thread too, so a follow-up after a
review still has the conversation in front of it.

---

## 6. Human review that actually resumes the graph

The graph compiles with `interrupt_after=["escalation_manager"]`. The first pass runs to the
escalation node, writes the queue row, and stops there with the checkpoint intact.

`POST /review/{request_id}` records the reviewer's decision, then:

1. Reads the checkpoint for that thread.
2. Writes the reviewed answer into `draft` and sets `reviewer_decision`, clearing
   `terminal_outcome` and `escalation_reason`.
3. Calls `graph.invoke(None, config)`, which resumes from the interrupt point and evaluates the
   conditional edge out of `escalation_manager`.

For `accept` and `edit` that edge routes to `output_guardrail`, so **the human's answer passes the
same PII scrub and the same "cite only what was retrieved" check as a machine-generated one**. A
reviewer who pastes in a clause that was never retrieved gets caught. A reviewer's answer is *not*
required to carry `[Document §clause]` markers: the reviewer is accountable for it, and demanding
citations blocked every plain-text edit. For `reject` the state is updated but the graph is not
resumed, nothing is released, and `GET /requests/{id}` reports `rejected` with no answer.

**The check runs before the decision is recorded.** Recording the decision is what stops two
reviewers releasing the same request, so it used to happen first - and an answer the guardrail then
refused left the request marked `reviewed` with nothing released: a second attempt got `409`, and
the asker never received an answer. Now `POST /review/{id}` runs the output guardrail on the reviewed
answer first (with the extracts stored in the review package) and answers `422` with the reason if it
would be refused, leaving the request in the queue so the reviewer can correct it and submit again.
If the resumed graph still refuses the answer (for example the PII check could not run), the request
is put back in the queue instead of being left half-reviewed.

The loop is bounded deliberately: if the guardrail rejects a reviewed answer,
`after_output_guardrail` sees `reviewer_decision` is set and routes to `END` rather than back to
escalation. Without that check the pair of conditional edges would cycle forever.

Checkpoints are keyed per request (`<thread_id>:<request_id>`). The resume only uses a checkpoint whose
`request_id` is the one under review - an escalation queued before per-request keys can find its chat's
checkpoint holding a later question's run. When no checkpoint of this request can be resumed (that case,
or the in-memory checkpointer after a restart), the answer that already passed the guardrail check is
released directly, recorded in the audit log, and the response says `resumed: false`.

---

## 7. The panel repair pass

`panel_repair_count` was declared in state and never used. The high-risk panel now gets exactly one
repair pass, bounded by `PANEL_REPAIR_PASSES`.

When the Challenger raises an objection it marks material and the consensus does not record it as
dissent, the panel re-runs consensus once with the objections fed back explicitly. The repair prompt
allows exactly three moves per objection: answer it from the evidence, record it as dissent in the
reviewer's words, or declare an unresolved conflict. Dropping an objection by ignoring it, and
weakening the answer into vagueness so the objection stops applying, are both ruled out — those are
the two failure modes a repair pass invites.

If the repaired consensus still leaves a material objection unaddressed, `unresolved_conflict` stands
and the request escalates.

---

## 8. Validation, confidence and the release gate

The validator produces a **defect list and never rewrites the answer**. A critic that can edit
patches claims instead of reporting missing evidence.

Deterministic checks run before the LLM validator: every citation in the answer must correspond to a
retrieved extract, and SQL evidence must pass `sanity_check` (row-cap truncation, empty results,
rows removed by scope, dates later than the as-of date).

**The clause-conflict heuristic was rewritten.** It used to flag any section heading that appeared in
chunks from more than one document. Compliance documents share headings constantly — *Purpose*,
*Scope*, *Definitions*, *Responsibilities* — and every image node is stamped with the section
`Figures`, so almost any answer drawing on two documents produced a `CLAUSE_CONFLICT`. That defect
type is both blocking and conflict-detecting, so it failed validation *and* tripped the confidence
gate, with no path past it at any risk level. It now requires all four of: a non-generic heading, at
least two chunks carrying obligation language (`must`, `shall`, `may not`, `required`), those chunks
coming from different documents, and different clause numbers. A real conflict between two binding
clauses still fires; a shared *Purpose* heading does not.

Confidence is four weighted signals — retrieval 0.30, validation 0.30, source agreement 0.20,
coverage 0.20, minus 0.10 when degraded. Risk is checked **independently** of confidence at the
release gate: High risk escalates at 0.97 confidence. Confidence measures whether the answer is
right; risk measures the cost if it is not.

---

## 9. Escalation reasons that name the real cause

`_derive_reason` used to report `"validation failed after N repair attempt(s)"` for every route into
the escalation node from validation — including the route taken when the budget guard refused to
start a repair. A request that escalated with zero retries used and two retries available produced
`"validation failed after 0 repair attempt(s)"`, which reads like a retry problem and is actually a
deadline problem. That single misleading string is what makes the 4-second deadline hard to find.

The three cases are now distinguished:

- retries genuinely exhausted → *"all 2 permitted repair attempt(s) were used up"*
- no time left → *"0.20s of the deadline remained and a repair needs at least 1s. This is a budget
  stop, not an exhausted retry count"*
- no tokens left → the same, with the token figures

A `budget_stops` entry from any node is reported ahead of the generic reasons, naming the node and
what ran out. The context package handed to reviewers carries `budget_stops` too.

---

## 10. State channels and reducers

`AgentState` is a `TypedDict` of LangGraph channels. **A field that is not declared as a channel is
silently discarded**, which is how two RBAC inputs went missing: `/ask` was setting
`state["departments"]` and `state["document_scope_request"]` on a state object whose schema declared
neither, so department filtering and caller-supplied document scope never reached the nodes that
read them. Both are now declared, along with the reviewer fields, `started_ts`, `marks` and
`budget_stops`.

`retrieved_chunks` uses a custom reducer. It was `operator.add`, and `reflection_node` returned `[]`
intending to clear it — but an accumulating reducer treats `[]` as "append nothing", so chunks
**accumulated across retries** instead of resetting. Every reflection pass therefore validated
against a growing pile of evidence from previous attempts, which made the clause-conflict heuristic
more likely to fire on the retry than on the first pass. `merge_chunks` appends a list and resets on
`None`, and reflection now returns `None`.

---

## 11. Auditing

`traced_node` writes one append-only `system_audit_log` row per node with the elapsed time, tokens
spent, remaining budget and outcome, plus a `budget_stopped` row when a node is refused. Terminal
nodes write their own richer rows on top (`answered`, `rejected`, `escalated`, `refused`).

`system_audit_log` carries `DO INSTEAD NOTHING` rules on UPDATE and DELETE, so the log is append-only
at the database level rather than by convention.

Audit writes are direct rather than buffered and flushed at the end. Buffering would be faster, but
an audit row that is lost when the process dies is worth less than the milliseconds it saves.

### The circuit breaker in front of the audit log

Direct writes put the database on the critical path of every node, and that is a deadline hazard.
`DB_CONNECT_TIMEOUT_SECONDS` is 3.0, so if Postgres is unreachable each audit write blocks for three
seconds before giving up. A request writes roughly a dozen of them, which is thirty-six seconds of
pure blocking against a thirty-second deadline: a database blip would budget-stop every request in
flight, with the escalation reason naming whichever node happened to be running. It is the same
failure shape as the four-second deadline, reached by a different route.

`AuditCircuit` in `src/core/audit.py` stops that. After three consecutive write failures it suspends
audit writes for thirty seconds, then lets one through to test the water; a success re-arms it. The
trade is deliberate and it is a real cost — the append-only log gets a gap for the length of the
window. It is the right way round because nothing in the answer or escalation path reads an audit
row, so blocking on the log buys no correctness and spends the budget that decides whether the
caller gets an answer at all.

The gap is made visible rather than silent. Opening the breaker logs an error naming the suspension,
recovery logs how many rows were lost, and `GET /health` reports `audit_log.writing` with the
consecutive failure count — so "the log has a hole here" is answerable after the fact.

---

## 10a. Reading a low score correctly

`confidence 0.485 is below the 0.75 release threshold (retrieval 0.0001, validation 0.95,
agreement 1.0, coverage 0.0)` is an accurate sentence that sends you to the wrong place. It reads
like a scoring problem. It is not one — retrieval at 0.0001 is a FlashRank cross-encoder score, and
a cross-encoder returning 0.0001 has read the passage and judged it unrelated to the question. The
pipeline searched, found nothing, and refused to invent. The defect is in the corpus, and the
sentence never says so.

`_derive_reason` now checks the best retrieval score before it reaches the confidence branch. Below
`RETRIEVAL_NO_MATCH_CEILING` (0.05), with no SQL evidence to lean on, it returns a different
sentence: that this is a no-match rather than a weak match, how many extracts came back, **which
documents were actually searched**, and that a document missing from that list is a document that is
not indexed. The list of titles is the part that matters — reading it and not finding the policy the
question is about is the whole diagnosis, and it takes a second.

The ceiling is deliberately low. Between 0.05 and the 0.75 threshold sits the genuinely weak match,
where the corpus does hold something relevant and the retrieval or the reranking is at fault. Those
must keep the confidence breakdown, because there the numbers are the diagnosis.

### document_scope is now rejected rather than ignored

`document_scope` narrows retrieval and can never widen it, so `build_scope_filters` discards
anything outside the role's grant and falls back to the whole grant when nothing survives. That is
correct behaviour and it is also silent — a caller sending `["string"]`, which is what the Swagger
example body contains by default, gets a full-scope search and no indication that the field was
thrown away.

`/ask` now validates the field against the caller's grant and returns 422 naming the offending
values and the ones the role accepts. A scope the role cannot read is refused the same way rather
than quietly downgraded, because "you asked for GDPR and got the privacy policy without being told"
is worse than an error. `effective_document_scope` filters against the grant as well, so the
inferred scope from intent classification cannot smuggle in a document type the role has no right
to — belt and braces, since the store filter enforces it too.

---

## 11a. The corpus bootstrap, and the check that used to hide a half-empty index

Startup calls `run_startup_bootstrap`. The version before this one opened with:

```python
if corpus_is_indexed():
    return None
```

where `corpus_is_indexed()` is `table_has_rows()` — **any row at all**. That is the wrong question. It
asks whether the table has ever been written to, and answers as if it had asked whether the corpus is
complete. Three stray rows left over from an unrelated experiment were enough to make every
subsequent startup skip the ingest and log `policy corpus already indexed` at INFO, while two of the
seven mandated policies sat unindexed. Questions those two answer retrieved nothing, scored ~0.0 on
retrieval and 0.0 on coverage, and escalated on low confidence — which looks like a scoring problem
and is not one.

Bootstrap now reconciles instead. `unindexed_files()` hashes each file in the corpus directory and
asks the vector table whether that hash is present — the same check `ingest_file` already makes per
file, so ingesting is idempotent and re-running costs seven hashes and seven index lookups. If
nothing is missing it returns without work. If something is missing it says which files, at WARNING,
and ingests. Afterwards it checks again, and anything *still* missing is logged at ERROR naming the
consequence, because a document that failed to parse is otherwise invisible until a user asks a
question about it.

The hash is the identity, not the title. Deriving an expected title from a filename works until a
document is called `Data_Retention_and_Archival_Policy.pdf` and the pipeline stamps it
`Data Retention and Archival Policy` while `str.title()` produces `Data Retention And Archival
Policy` — one capital letter, and the reconciler decides a fully indexed document is missing.

### Rebuilding

Reconciliation adds what is absent. It cannot remove what should not be there — a stray row is
indistinguishable from a legitimate one, and deleting on a guess is worse than leaving it. When the
table holds documents that do not belong to the corpus, `POST /ingest/rebuild?confirm=true`
(corpus-admin only) truncates the vector table and re-ingests the directory from scratch.

`confirm` is required because the call costs a full embedding pass over the corpus and destroys the
existing vectors; a rebuild started by a mis-click is expensive and not undoable. If the truncate
succeeds and re-ingestion then fails, the response says exactly that rather than reporting a generic
500 — the difference matters, because at that point the index is empty and every question will
escalate until it is fixed.

`GET /ingest/status` now also returns `unindexed_files`, so the gap is visible before a user finds it.

---

## 12. Running it

```bash
uv sync
python scripts/init_db.py          # applies data/sql/schema.sql
uvicorn main:app --reload --port 8000
```

`scripts/init_db.py` reads `data/sql/schema.sql`, which now also creates `conversation_turns`,
`conversation_threads` and `request_latency`. The root-level `sql/schema.sql` is an older partial
copy and is not what the initialiser reads.

### Settings that matter

| variable | working value | what goes wrong below it |
|----------|---------------|--------------------------|
| `DEADLINE_SECONDS_STANDARD` | 30.0 | everything escalates before it is scored |
| `DEADLINE_SECONDS_HYBRID` | 45.0 | hybrid degrades to rag |
| `DEADLINE_SECONDS_AGENTIC` | 75.0 | agentic degrades to hybrid |
| `DEADLINE_SECONDS_HIGH_RISK` | 90.0 | the panel cannot finish |
| `DEFAULT_TOKEN_BUDGET` | 24000 | validation is skipped; below 16000 the floor raises it and warns |
| `CONFIDENCE_THRESHOLD` | 0.75 | lowering this hides a scorer that stopped discriminating |
| `SLO_PATH_TARGET_RATIO` | 0.85 | at 1.0 the target restates the deadline and never warns early |
| `CONVERSATION_REWRITE_TURNS` | 5 | the rewrite loses the referent a follow-up depends on |
| `PII_ENTITIES` | a bare comma-separated list | a malformed value silently disables redaction for the first entity in it |
| `MODEL_ROUTING_STRATEGY` | `heuristic` | `static` restores the fixed strong tier for every generation call and the cost report flattens |
| `SEMANTIC_CACHE_THRESHOLD` | 0.95 | lower and two different questions start sharing an answer; 1.0 is exact-match only |
| `ROUTING_LATENCY_PRESSURE_SECONDS` | 8.0 | under it a routable node drops to the small tier; 0 disables the latency downgrade |
| `COST_BUDGET_USD_PER_REQUEST` | 0.05 | a soft ceiling: exceeding it is logged and counted, never blocked |

Changing `.env` needs a restart — `uvicorn --reload` watches `.py` files, not `.env`.

### Calling it

Tokens are minted by **Auth0**, not by this backend (RBAC follows the practice-1 travel planner
method: `src/auth/security.py` validates the RS256 JWT against the tenant's JWKS and reads the
roles the `add-roles-to-tokens` Action put under `https://stateful-agent.com/roles`). The React
frontend in `../frontend` does this for you; for curl / Swagger get an access token from Auth0 first
(e.g. log in through the frontend and copy it from the browser devtools, or use the Auth0 API
explorer with the `https://api.stateful-agent.com` audience).

```bash
TOKEN="eyJ..."   # Auth0 access token for a user with one of the five roles

# 1. who am I according to the backend: role, scopes, screens
curl localhost:8000/auth/me -H "authorization: Bearer $TOKEN"

# 2. hand over the Azure OpenAI credentials typed on the login page (any role)
curl -X POST localhost:8000/auth/azure -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"endpoint":"https://<resource>.openai.azure.com/","api_key":"...","small_deployment":"gpt-4o-mini","strong_deployment":"gpt-4o","embedding_deployment":"text-embedding-3-small"}'

# 3. ask; document_scope is optional and only ever narrows
curl -X POST localhost:8000/ask -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"query":"What is the retention period for customer transaction records?"}'

# 4. follow up in the same thread by passing back the thread_id you got
# 5. reviewer endpoints
curl localhost:8000/review/queue        -H "authorization: Bearer $TOKEN"
curl localhost:8000/metrics/slo         -H "authorization: Bearer $TOKEN"
```

In the Swagger page, click **Authorize** and paste the Auth0 access token on its own — Swagger adds
the word `Bearer` itself.

Two things to know when testing. Repeating an identical query that was *answered* (`200`) returns the
same response for 15 minutes without running the graph, because `build_key` falls back to hashing the
request body when no `Idempotency-Key` header is present; change a word or send a key. A `202
pending_review` is only replayed for a retry that sends the same `Idempotency-Key` - without a key the
question runs again, so asking again after a review does not come back as the old pending body. And a `202` body always carries a
`reason` that names the exact gate that fired, which is the fastest way to find out why something
escalated.

## 13. The clause identifier bug that made `/ask` never return 200

For a while every answerable question came back as `202 pending_review` with no citations. The
escalation reason always named the budget guard, which made it look like a budget problem. It was
not. It was a metadata problem two layers upstream, and the budget was only where it surfaced.

### What was actually wrong

`CLAUSE_HEADING` in `src/ingestion/extractors/clause_extraction.py` matched a numbered markdown
heading, but it required whitespace directly after the number:

```
^(#{1,4})\s*(?:§\s*)?([0-9]+(?:\.[0-9]+)*|[A-Z]\.[0-9]+(?:\.[0-9]+)*)\s+(.+)$
```

The policy PDFs come out of LlamaParse as `## 1. Purpose & Scope` — number, then a period, then the
title. After `1` the next character is `.`, not whitespace, and `(?:\.[0-9]+)*` cannot consume it
because `Purpose` is not a digit. So the heading never matched. Only headings written `## 2.1 User
Registration`, with no trailing dot, matched — which is why the ISO 27001 summary looked healthy and
nothing else did.

When no heading matches, `extract_clauses` falls back to one section for the whole document, headed
`General`, numbered `1`. That is the failure: a nine-section policy became a single clause. The
vector table showed it plainly — every Supplier & Vendor chunk carried `clause_number = 1`, and every
GDPR chunk did too.

### Why a metadata bug came back as a budget error

The chunk body still contained the real headings, so the generator read `## 3. Vendor Onboarding &
Due Diligence` inside the extract and cited `[Supplier Vendor Compliance Policy §3]` — correctly, as
far as it could tell. But `_deterministic_defects` builds the set of known clauses from chunk
*metadata*, which only ever said `§1`. So the citation was not in the known set, the validator raised
`ungrounded_claim`, that defect is blocking, and the graph went to `reflection`.

Reflection clears the evidence and returns to the planner, so the whole path re-ran and produced the
same citation against the same broken metadata, and failed the same way. Three rounds of that spend
more than the token budget allows, so the third `compliance_validation` was refused by the budget
guard and the request escalated. The budget guard was the messenger.

This is the same shape as the two earlier escalate-everything bugs: a check that is correct in
isolation, fed an input that is wrong in a way the check cannot see.

### The fix

The number may now be followed by an optional period, and `Section`, `Clause` and `Article` are
accepted as prefixes, which is what recovered the GDPR articles as `§5` and `§32` instead of one
undifferentiated `§1`. Headings are run through `html.unescape` because LlamaParse emits `&#x26;`
where the document says `&`, and that entity was reaching citations and section labels.

`splitters.py` had its own copy of the same regex. It now imports the one in `clause_extraction`, so
the two cannot drift apart again.

Re-ingestion is required after any change here — the clause number is stamped at ingestion time, so
fixing the parser does nothing to chunks already in the table. `POST /ingest/rebuild?confirm=true`.

### Four defects found alongside it

**The chunk id was reachable as a citation.** `format_context` wrapped each extract with
`neutralise_retrieved(..., source=chunk.chunk_id)`, which renders as
`<retrieved_document source="69c5e1d2-...">`. The generator treated that identifier as citable and
emitted `[Supplier Vendor Compliance Policy §69c5e1d2-aafb-43ec-bad2-975c479b5d0b]`. The wrapper now
carries the citation instead of the internal id, so the only identifier visible to the model is one
it is allowed to use.

**Citations were built three different ways.** `RetrievedChunk.citation`, `format_context`,
`_deterministic_defects` and `_citations_from` each formatted `title §clause` themselves, and each
produced `Doc §` with a trailing marker for image nodes, which carry no clause. The property is now
the single source of truth and drops the marker when there is no clause number; the other three call
it. `_citations_from` also compares through `canonical_citation` and de-duplicates, which it did not
do before — a draft citing the same clause in three sentences returned that citation three times.

**`policy_record_conflict` could fire with no records to conflict with.** The LLM validator was
raising it on pure policy questions, reasoning that the absence of rows meant practice could not be
confirmed. A conflict between a policy and a record is not possible when no record was read, so
`policy_record_conflict` and `sql_sanity_failure` are now dropped when `sql_evidence` is None. The
observation the validator was making is a coverage gap, which it already reports separately and which
does not block release.

**The scope guard refused valid questions on plurals.** `is_in_domain` compared exact tokens against
`IN_DOMAIN_TERMS`. "Which vendors are currently non-compliant" tokenises to `vendors` and `compliant`
— the set contained `vendor` and `compliance`, so the question was refused outright in 20ms as having
no compliance subject. Tokens are now reduced to their singular before the comparison, and the term
list gained the words that were missing. Out-of-domain questions are still refused; the markers for
those are unchanged.

**Generic plurals were resolved as entity names.** With the scope guard fixed, "which vendors are
non-compliant" reached entity resolution, which fuzzy-matched the common noun `vendors` against
vendor *names* and asked which of `Vendor_1 … Vendor_4` was meant. A class noun denotes the class, not
an instance, so `GENERIC_ENTITY_SURFACES` is skipped before resolution and the question goes to the
aggregate template path where it belongs.

### Swagger showed one response shape when `/ask` returns four

`/ask` had no `response_model` and no `responses`, so FastAPI documented a bare `200` with an empty
schema and no `202` at all — the endpoint that returns four different bodies documented none of them.
`response_model` is deliberately still unset, because the handler returns already-dumped dicts and
attaching a model would re-validate them on the way out. Instead `ASK_RESPONSES` declares the shapes:
`200` as a discriminated union of `AnswerResponse`, `RefusalResponse` and `ClarificationResponse` on
the `status` field, and `202` as `PendingReviewResponse`. Swagger now shows all four, and `status` is
documented as the field that tells them apart.

### The budget could not fund the retries the config allowed

`MAX_REFLECTION_RETRIES` is 2, so a request may run three evidence rounds. On the hybrid path a round
costs 800 + 4000 + 2500 + 900 = 8200, and the preamble costs 2100, so three rounds need about 25800
before the panel path is considered at all. `DEFAULT_TOKEN_BUDGET` was 24000. The last retry could
therefore never reach a verdict — it always died on the budget guard instead of on its own merits,
which is why the escalation reason named the budget for failures that had nothing to do with it. The
budget is now 34000, which covers three rounds of the most expensive path.

### What is still open

The reflection loop remains the only thing standing between a valid question and a `200`. It is
expensive by construction: any blocking defect re-runs the entire evidence path, so a `missing_citation`
— a drafting problem, where the retrieved evidence was fine — is repaired by re-retrieving the same
chunks and re-drafting, at 8200 tokens and roughly ten seconds a round. Three rounds exhaust either
the token budget or the 30-second standard deadline, and the request escalates on the guard rather
than on a verdict.

Two changes would address it, and both are design decisions rather than repairs:

- Route a defect that is about the *draft* rather than the *evidence* to a re-draft against the
  chunks already in state, instead of through the planner. `missing_citation` and `ungrounded_claim`
  do not need new evidence.
- Stop the loop before it starves. `after_validation` could escalate as soon as the remaining budget
  cannot fund another complete round, which turns a mid-step budget kill into an honest early
  escalation and saves the twenty seconds the doomed round would have spent.

Separately, the planner routes pure policy questions to `hybrid` more often than the question
warrants. The `rag` path answers them in about eleven seconds against roughly thirty for `hybrid`,
and the hybrid prompt has to be told what to do with an empty result set — it now is, but not
choosing that path for a question with no record component would be better than handling it well.


## 14. The lexical risk trigger that was matching substrings

`lexical_risk_floor` in `src/guardrails/scope.py` walks a small list of keywords and returns the
highest risk floor any of them justifies. The list has to be short because it runs on every request
before the L2 classifier does, and cheap lookups are the only reason having an L1 layer at all is
worth it — a slow lexical check would just be a worse classifier.

The first version compared each keyword with `in` — plain Python substring containment. That
worked for the phrase keywords (`"personal data breach"`, `"right to be forgotten"`) but silently
misbehaved for the short single-word ones. Three examples that were all firing the wrong floor:

- `"bribe"` lives in the High list, so any query mentioning the anti-bribery policy fired High.
  That routed the request to the multi-agent panel, widened the deadline to ninety seconds, and
  let three parallel LLM agents drain the token budget before compliance_validation could run.
  The response came back as `202 pending_review` with `risk_level: "High"` and a budget-exhausted
  reason — none of which had anything to do with the actual question.
- `"dispose"` lives in the Medium list, so a query asking about a `disposal_date` field on a
  retention record fired Medium and was routed through the more expensive path.
- `"sanction"` would have caught `"sanctioned"` in a supplier context if the corpus used that word.

The fix is to match on word boundaries instead of raw substring:

```python
if re.search(rf"\b{re.escape(keyword)}\b", lowered):
    matched.append(keyword)
```

`\b` sits between a word character and a non-word character, so `\bbribe\b` matches "a bribe was
offered" but not "anti-bribery" — because the `y` after `bribe` is a word character and there is
no boundary there. Hyphenated keywords like `"non-compliant"` keep working, because `-` is a
non-word character and the boundaries sit at the outside of the whole phrase, not around the
hyphen. Multi-word phrases like `"personal data breach"` keep working for the same reason — the
spaces inside are non-word characters and the required boundaries are at the outer edges.

`re.escape` is there so a future keyword that contains a regex metacharacter — `+`, `.`, `(` —
cannot break the pattern. The keyword list is data, and the matcher should not care whether the
data happens to look like regex.

Six tests in `tests/test_guardrails.py` pin the new behaviour: anti-bribery policy lookups return
Low, a standalone `bribe` still returns High, `disposal date` returns Low, `dispose of records`
returns Medium, and `non-compliant` still returns Medium. When someone adds a new keyword, running
those tests is the check that the new addition does not shadow a common English fragment the way
`"bribe"` shadowed `"anti-bribery"`.

### Why this reads out as a budget error rather than a risk-trigger error

The 202 body only says the budget guard stopped compliance_validation, because that is the last
thing the request tried to do. Nothing in the response says the request was routed through the
multi-agent panel, and nothing says which lexical trigger fired the High floor. The only way to
see the misclassification today is to read the Langfuse trace or the append-only audit rows.

Widening the escalation reason to include the risk signals — specifically the L1 matched keywords
when they raise the floor above Low — would surface bugs of this shape without leaving the
response body. That change belongs in `_derive_reason` (`src/nodes/escalation.py`) and is listed
alongside the other known escalation-reason improvements in section 9.

### Related budget change

`DEFAULT_TOKEN_BUDGET` was raised from 34000 to 48000 in `.env`. The old floor was enough for RAG
and NL2SQL but tight for the panel, and the failure mode was the same shape as the substring bug
above — a request that reached the panel legitimately would blow the budget just as surely as one
routed there by mistake. The 48000 headroom is chosen so a full panel run with one repair pass
still leaves room for compliance_validation and confidence scoring. The `MINIMUM_DEADLINES`
validator in `configs/settings.py` still rejects anything under 16000, so this can be lowered
again but not below the floor.

---

## 15. Path selection: making the router deterministic

This section covers the pass that fixed two related complaints — that the choice between RAG,
NL2SQL, hybrid and agentic was inconsistent, and that questions about records came back with mixed
or missing answers.

### 15.1 What was actually wrong

Four separate defects produced the same symptom.

**The planner computed a path from the intent and then ignored it.** `INTENT_DEFAULT_PATH` mapped
each intent to a path, but that mapping was only consulted inside the `except` branch, as a fallback
for when the model call failed. On the success path the LLM's own choice of `plan.path` was used
verbatim. Nothing checked it against the intent. So the same question, asked twice, could route to
`rag` once and `hybrid` the next time, and a `record_lookup` could be answered from policy text —
which is the "mixed answer" the user sees, because policy text cannot state what a record holds.

**Degradation dropped the evidence the question turned on.** Under deadline pressure the router
degraded `hybrid` to `rag` unconditionally. For "which vendors are overdue for review", `rag` has no
chance of answering: the answer is a row. The request would either produce a hedge or fail
validation and escalate. Degrading has to preserve the source the question actually needs.

**An honest empty SQL result was treated as a defect.** `sanity_check` returned one list mixing two
different kinds of thing: caveats the narration must disclose, and genuine problems. The narration
node used it correctly, as caveats. The validator then re-ran the same function and turned every
entry into a `SQL_SANITY_FAILURE`, which is a blocking defect. So a query returning zero rows failed
validation, went to reflection, re-ran, returned zero rows again, and escalated. A correct answer —
"no vendor is overdue as of 2025-12-31" — could never be released. The same applied to any result
narrowed by the role's department or risk-band scope.

**A forward-looking date was read as a data error.** The as-of check flagged any column whose name
ended in `_date` with a value after the pinned date. `target_resolution_date` ends in `_date` and is
supposed to be in the future for any open finding. Every open finding with a future deadline minted
a blocking defect.

Two more things fell out of the same review:

**Reflection's re-plan was thrown away.** `reflection_node` cleared `validation` before returning,
and `planner_node` built its revision context from `state["validation"]`. By the time the planner
ran, that was `None`, so the re-plan prompt carried no defect information and the model produced the
same plan that had just failed. Two retries later the request escalated. Reflection now hands the
planner its revised plan directly, along with a written directive naming the defects.

**BM25 ignored the document scope.** `lexical_search` accepted a `doc_scope` argument and never used
it. Dense retrieval narrowed to the requested scope and the lexical half did not, so fusion could
surface a document the caller had explicitly scoped out — within the role's grant, but not what was
asked for.

### 15.2 The decision table

`src/graph/path_policy.py` is now the single place a path is chosen. It is a pure module: no model
call, no I/O, fully unit-tested. Every rule below is one row of `RULES`.

| Intent | Default path | Paths the planner may choose | Evidence the answer turns on | Degrades to | Why the set is this narrow |
|---|---|---|---|---|---|
| `policy_lookup` | `rag` | `rag` | policy | — | The answer is a clause. The database holds no policy text, so no other path can produce it. |
| `record_lookup` | `nl2sql` | `nl2sql` | records | — | The answer is a row. Policy text cannot state what a record holds. |
| `vendor_status` | `nl2sql` | `nl2sql`, `hybrid` | records | `nl2sql` | Status is a stored column. Policy only enters when the question also asks whether that status is permitted. |
| `retention_query` | `hybrid` | `rag`, `nl2sql`, `hybrid` | both | `nl2sql` | "How long do we keep X" is policy; "what is overdue" is a record. The question can be either, so all three are admissible. |
| `compliance_check` | `hybrid` | `hybrid` | both | `hybrid` | This intent *is* a rule checked against a record. Answering from one side alone is a wrong answer, not a partial one. |
| `incident_guidance` | `hybrid` | `rag`, `hybrid` | both | `rag` | What to do is written in policy; a record only sharpens it when the incident names one. (Was `agentic`, which the v4 workflow does not have.) |
| any, when risk fuses to `High` | `high_risk_panel` | — | both | — | Mandatory. The panel is not a path the planner may decline. |

The planner still writes the evidence plan — the steps, the objectives, the `must_prove` statements
that the validator later checks the answer against. What it no longer does is choose freely from all
five paths. Its proposal is checked against the admissible set; a proposal outside the set is
replaced with the intent's default and the substitution is recorded in `path_decision` with the
reason. The prompt tells the model the set up front and tells it plainly that naming a path outside
the set throws its reasoning away rather than widening what the system does.

### 15.3 Capability gates

Two checks run after the clamp, both deterministic, both in `_apply_capability_gates`.

A path that reads records, for a role granted no table: if the question turns only on records, the
route is `no_access` and the reply says plainly that this role cannot read the source that answers
it. It does not quietly fall back to RAG, and it is not queued for review either — a reviewer must not
hand a role data it may not read. A `store_associate` asking "is Sable Analytics compliant" gets "I
can't answer this with your access…", not a policy essay that never answers the question. (A user who
explicitly asks for a human still gets one.) If the question turns on both sources, it falls back to
the half that is available and sets `partial_evidence`, which costs confidence and has to be disclosed.

A path that reads documents, for a role granted none: the mirror image.

### 15.4 Degradation preserves the load-bearing source

The old ladder was `agentic → hybrid → rag`, regardless of the question. The new one reads the
intent's `fallback`:

- `agentic` under `AGENTIC_MIN_SECONDS` degrades to `hybrid`, which still reads both sources.
- `hybrid` under `HYBRID_MIN_SECONDS` degrades to whichever single source the intent turns on —
  `nl2sql` for `retention_query` and `vendor_status`, `rag` for `incident_guidance`.
- `nl2sql` and `rag` do not degrade. They are already the cheapest path that can answer their
  question, so there is nothing below them but a wrong answer. Under pressure they escalate.

`HYBRID_MIN_SECONDS` is a new setting, defaulting to 2.0. The old code had `1.5` written inline.

### 15.5 Disclosures versus defects

`src/sqlpath/disclosure.py` splits what `sanity_check` used to conflate.

A **disclosure** is something the answer must say. Zero rows, a truncated result, rows removed by the
role's scope. Each disclosure carries a regex that recognises the disclosure having been made. The
validator raises a defect only when a required disclosure is *absent from the answer text*. So:

- "The query returned no rows as of 2025-12-31; an empty result is not evidence of compliance"
  passes, and is released.
- "All vendor reviews are up to date" against the same zero-row result fails, because it states as
  fact something the empty result does not support.

That is the behaviour the system was always supposed to have. The old code could not express it,
because it had no way to tell a caveat from a defect.

A **data defect** is a genuine problem in the rows, and always blocks. There is one:
an *observed* date after the pinned as-of date. `OBSERVED_DATE_COLUMNS` names them explicitly —
`onboarding_date`, `last_audit_date`, `last_review_date`, `review_date`, `issue_identified_date`,
`resolution_date` — rather than guessing from the `_date` suffix. Forward-looking columns
(`next_review_due`, `target_resolution_date`) are expected to be in the future and are not checked.

Both the hybrid path and the high-risk panel's Data Verifier now receive the caveat list too. They
did not before, so a hybrid answer over an empty or capped result had no way of knowing it needed to
disclose anything, and then failed the check for not disclosing it.

### 15.6 Confidence for a records-only answer

Two scoring defects kept correct SQL answers below the 0.75 release threshold.

`_agreement_score` returned 0.7 whenever the plan expected records — including when the records were
present. The penalty is meant for the opposite case: the plan expected records and the answer only
has policy. It now fires only when the expected records are actually missing.

`_retrieval_score` returned a flat 0.5 for any answer with no retrieved chunks, which is every
NL2SQL answer. That is not a measurement, it is a handicap, and it put a clean SQL answer at
approximately 0.75 — exactly on the threshold, where a single caveat sank it. The score is now
computed from the SQL evidence itself: 0.90 for an executed template that returned rows, 0.75 for
one that honestly returned none, minus 0.10 each for truncation and for scope filtering.

A clean NL2SQL answer now scores about 0.93. An honest empty result scores about 0.89 and is
released with its caveat. Both are pinned by tests in `tests/test_sql_disclosure.py`.

### 15.7 Where the decision shows up in the logs

The routing decision is not inferable from the response, so it is written in three places.

`planner_node` emits one INFO line per request:

```
route intent=record_lookup proposed=rag chosen=nl2sql clamped=True replanned=False
      reason=intent record_lookup admits ['nl2sql']; the planner proposed rag, so it was
      clamped to nl2sql because the answer is a row; policy text cannot state what a record
      holds request_id=...
```

`path_decision` goes into state, so `_span` carries `routed_path`, `route_reason` and
`route_clamped` into the trace and the Langfuse span. `traced_node` builds the span from the state
merged with the node's result, so a value the node just set is visible on the node's own span rather
than the next one's.

The audit row for the planner node carries the same detail, which means a request that took an
unexpected path can be explained after the fact from `system_audit_log` alone, without a Langfuse
account.

`tests/test_route_logging.py` asserts the log line and the span fields, so this stays true.

### 15.8 The seed file did not match the schema

`data/sql/seed.sql` inserted `vendors.department`, `vendors.category`,
`vendors.handles_personal_data`, `vendors.contract_start_date`, `retention_records.record_type`,
`compliance_reviews.outcome` and several more columns that do not exist in `data/sql/schema.sql`. It
was left over from an earlier schema and could not run. Its status values were in the wrong
spelling too — `non_compliant` where the templates bind `Non-Compliant`, which returns zero rows and
reads as "no problems found".

It has been rewritten against the current schema: 12 vendors, 13 audit findings, 18 retention
records and 13 compliance reviews, with every enum value in the exact stored spelling, and dated so
that the 2025-12-31 as-of date produces a mix of overdue, due and current rows. It is deterministic,
so an eval run is reproducible. `generate_capstone_sql_data.py` remains available for a larger
random dataset.

### 15.9 Ruff

`pyproject.toml` had a `[tool.ruff]` section with a line length and nothing else, so `ruff check`
enforced only the default `E`/`F` subset. It now selects `E`, `F`, `W`, `I`, `UP`, `B`, `C4`, `SIM`,
`RET`, `ARG`, `PTH`, `T20` and `RUF`, and the tree is clean under it.

Three ignores are deliberate and worth knowing about:

- `UP042` — the enums stay `class X(str, Enum)` rather than `StrEnum`. `StrEnum` changes what
  `str(member)` returns, and these values cross a pydantic boundary and a database boundary. The
  change is safe in principle and not worth the risk here.
- `ARG001` in `src/api/routes.py` and `main.py` — a `principal: Principal = Depends(...)` argument
  that is never read is not dead code. It is what makes FastAPI run the auth dependency. Removing it
  removes the authorisation check.
- `RUF005` in the three PyMuPDF extractors — `Rect(...) + (a, b, c, d)` is PyMuPDF rectangle
  arithmetic, not list concatenation. Ruff's suggested rewrite would break it.

Everything else was fixed rather than silenced, including 14 real `B904` cases where an exception
raised inside an `except` block lost its cause chain.

---

## 16. The generated NL2SQL engine behind the vetted catalogue

Section 4 describes the vetted templates. This section describes what happens when none of them fit,
which used to be a 202 and is now an answer.

### 16.1 The failure this fixes

A `record_lookup` asking to list the whole vendor roster escalated with:

> the records path produced no evidence: The catalogue does not provide a query to list all vendors
> by their approval status.

The selector was right. Of the six `vendors` templates, five require a filter argument
(`vendor_id`, `name_fragment`, or one of the status/band enums) and the sixth,
`vendors_review_overdue`, filters on the review date. **No template returns an unfiltered roster.**
Only six of the seventeen templates can be called with no argument at all. The catalogue was written
around the questions we anticipated, and "show me everything" was not one of them.

Routing that question to the ReAct agent instead would not have helped: `compliance_records`, the
agent's SQL tool, calls the same `run_vetted_sql`, so it would have hit the identical refusal and
returned "The database query was refused: …" to the agent, which would then either give up or invent
an answer. A fallback is only a fallback if it uses a different engine.

### 16.2 Two tiers

`run_vetted_sql` is now a two-tier read.

**Tier 1 — the catalogue.** Unchanged. A reviewed query, typed parameters, bound through psycopg.
When a template fits, this is what runs and nothing below happens.

**Tier 2 — the generated engine.** `src/sqlpath/nl2sql_engine.py`. It runs on exactly one trigger:
`TemplateSelectionError`, meaning the catalogue genuinely cannot answer the question. Every other
failure still raises as before — a blocked question, a binding error, an out-of-scope statement, a
database fault. A question the guardrails refused never reaches the generator.

The engine:

1. Builds a LlamaIndex `SQLDatabase` over the SQLAlchemy engine with `include_tables` set to the
   role's grant, and takes the schema text from it. RBAC is applied by construction — the model is
   never shown a table the caller cannot read, so it cannot name one.
2. Generates one statement with `NL2SQL_GENERATION`, which supplies that schema, the domain rules
   from `SCHEMA_NOTES`, and the as-of date as a literal to write inline.
3. Puts the generated SQL through the same gates the templates pass: `validate_generated_sql`
   (starts SELECT or WITH, no write verbs, no stacked statements or comments, no clock read) and
   `assert_tables_in_scope` (every FROM and JOIN inside the grant), plus a placeholder check —
   a statement still carrying `%(name)s` or `:name` is rejected rather than run.
4. Executes it through the same `read_only_connection()` as the templates, so the read-only
   transaction, the statement timeout and the row cap all still apply.
5. Returns `SqlEvidence` with `generated=True`, then the caller applies `_scope_rows`, so department
   and risk-band narrowing works exactly as it does for a vetted result.

Generation and execution are deliberately separate. `NLSQLTableQueryEngine` would do both, and
letting it execute would mean losing the read-only transaction, the timeout, the row cap and the
scope filter. The engine writes the SQL; this codebase decides whether it runs.

### 16.3 A generated answer has to say so

`generated=True` adds a disclosure, on the same footing as an empty result or a row cap: the answer
must state that no reviewed query covered the question and the SQL was generated for it. If the
answer does not say it, the validator raises a blocking `SQL_SANITY_FAILURE` and the answer goes back
for a repair pass. Confidence also takes a 0.15 haircut, so a generated answer scores below the
equivalent vetted one and lands nearer the escalation threshold.

That is the honest position. The gates can prove a generated query is *safe* — it cannot write, it
cannot read outside the grant, it cannot read the clock. Nothing can prove it is *correct*: a query
reading `vendors.risk_category` for a question about `audit_logs.issue_severity` passes every gate
and returns a confident wrong answer. So the answer discloses which tier produced it and the reader
decides how much weight to give it.

Set `ENABLE_GENERATED_SQL_FALLBACK=false` to restore the previous behaviour, where a catalogue miss
escalates. The vetted-only guarantee is one environment variable away, and evaluation runs that need
it should set it.

### 16.4 `NOW()` was never being caught

Fixing this turned up a live bug in `CLOCK_READ`. The pattern ended in `\b` after the group, and for
`NOW()` the group ends on `)` — a non-word character with nothing after it, so the boundary never
matched and `NOW()` passed the check. `CURRENT_DATE` matched, which is why it went unnoticed.

The pattern is now split: word-boundary alternatives for the bare keywords, call-shaped alternatives
for the functions, and it covers `LOCALTIME`, `CURRENT_TIME`, `TRANSACTION_TIMESTAMP`,
`STATEMENT_TIMESTAMP`, `CLOCK_TIMESTAMP` and `TIMEOFDAY` as well. Ten forms are pinned by
`tests/test_generated_sql.py`, along with three that must not false-positive — including a column
named `now_status` and a `LIKE '%know%'`.

This mattered little while every statement was hand-reviewed. It matters a great deal now that a
model writes them: "which vendors are overdue" is exactly the question a generator answers with
`next_review_due < NOW()`, and that would have silently reported against today's clock instead of the
pinned 2025-12-31, making the answer wrong and unreproducible.

### 16.5 The catalogue is still worth extending

The fallback is a safety net, not a reason to stop adding templates. A vetted query for a common
question is faster, cheaper, correct by review, and releases at higher confidence with no disclosure.
The obvious gaps this failure exposed: a full vendor roster, a `GROUP BY` count of vendors per
status, all open findings, and all retention records. Add those and the four most likely "show me
everything" questions never reach tier 2 at all.
mohit
## 17. API reference — every endpoint

Base URL: `http://localhost:8000` · Interactive docs (Swagger): `http://localhost:8000/docs`

**One rule that catches everyone:** every endpoint except `GET /health` requires an **Auth0**
JWT (RS256, audience `API_AUDIENCE`). Log in through the frontend (or the Auth0 API explorer), then
send the access token as `Authorization: Bearer <token>` on every call. In the Swagger page it is
*not* attached automatically — click **Authorize** and paste only the token, without the word
`Bearer`.

Shell snippets below assume you have saved a token:

```bash
TOKEN="eyJ..."   # an Auth0 access token
```

### 17.1 `GET /health` — no auth
Liveness plus DB state, audit-log circuit status, escalation queue depth, and the active
`confidence_threshold` / deadlines.

```bash
curl http://localhost:8000/health
```

### 17.2 `GET /auth/me` — who the token is
Returns the role the Auth0 token resolved to, its access scopes (`doc:*`, `table:*`, `risk:*`),
`departments`, the raw `auth0_roles`, the `permissions` flags and the `screens` the frontend may
open. The role names in Auth0 must be exactly
`store_associate`, `store_manager`, `compliance_officer`, `legal_reviewer`, `admin`
(`user` from practice-1 maps to `store_associate`; several roles → the most privileged wins;
no recognised role → `403`).

```bash
curl http://localhost:8000/auth/me -H "Authorization: Bearer $TOKEN"
```

### 17.2a `POST /auth/azure` — Azure OpenAI credentials from the login page
The frontend never keeps the Azure key in a `.env`; the user types it on the login page and the
frontend sends it here once after the Auth0 login. With `verify` (default `true`) the backend makes one
tiny embedding call and one tiny chat call first and answers `400` if either deployment rejects the
key, otherwise it swaps the values into the running settings and drops every cached model client
(`src/auth/azure_credentials.py`). `GET /health` reports `azure_configured` afterwards.

```bash
curl -X POST http://localhost:8000/auth/azure \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"endpoint": "https://<resource>.openai.azure.com/", "api_key": "...", "api_version": "2024-10-21",
       "small_deployment": "gpt-4o-mini", "strong_deployment": "gpt-4o", "embedding_deployment": "text-embedding-3-small"}'
```

### 17.3 `POST /ask` — the main endpoint
Body is `AskRequest`: `query` (required, 3–2000 chars), `thread_id` (optional, to continue a
conversation), `document_scope` (optional, only ever *narrows* what the role can already read).
Optional header `Idempotency-Key: <any-string>` makes a retry return the first result instead of
re-running.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "List all vendors in the Low risk category."}'
```

Two possible status codes:
- **`200`** — a terminal result. Read the `status` field: `answered` (answer + citations),
  `refused` (out of domain), or `clarification_required` (an entity was ambiguous or unresolved).
- **`202`** — `pending_review`. The system could not certify an answer and queued it for a human;
  no draft is returned. Poll the `poll_url` it gives you.

**Cache behaviour (section 18.2).** A question that a caller in the *same access scope* (role,
scopes, departments, `document_scope`) already had certified is answered from the response cache:
same body, a fresh `request_id`, `timings.total_ms` in single-digit milliseconds, and a `cache`
block naming the kind of hit (`exact` / `semantic`), the similarity, the age and the
`source_request_id`. Pass `"use_cache": false` to force a graph run. Follow-ups that carry
conversation history are never served from cache. Every response's `trace` now also carries
`model_routing` (which tier each routable model call used, and why) and `llm_usage` (the tokens
the provider actually billed and their USD cost).

### 17.4 `GET /requests/{request_id}` — poll an escalated request
After a `202`, take the `request_id` and poll here until a reviewer acts.

```bash
curl http://localhost:8000/requests/<request_id> \
  -H "Authorization: Bearer $TOKEN"
```

Returns `pending_review` while it waits, then `answered` with the released answer (the text that passed
the output guardrail, PII scrubbed) and `reviewer_decision` (`accept` or `edit`), or `rejected` with a
`note` and no answer when the reviewer rejected it.

### 17.5 `GET /review/queue` — reviewer only
Everything currently waiting for review. Needs `compliance_officer`, `legal_reviewer`, or `admin`.

```bash
curl http://localhost:8000/review/queue -H "Authorization: Bearer $TOKEN"
```

### 17.6 `GET /review/{request_id}` — reviewer only
The full context package for one escalated request: original query, retrieved documents, SQL
evidence, validation output, reasoning trace, draft answer, risk and confidence.

```bash
curl http://localhost:8000/review/<request_id> -H "Authorization: Bearer $TOKEN"
```

### 17.7 `POST /review/{request_id}` — reviewer submits a decision
Body is `ReviewSubmission`: `decision` is `accept`, `edit`, or `reject`; `edited_answer` (used with
`edit`); `reviewer_notes` (optional).

```bash
curl -X POST http://localhost:8000/review/<request_id> \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"decision": "edit", "edited_answer": "Corrected answer text.", "reviewer_notes": "fixed clause ref"}'
```

`accept`/`edit` are checked by the output guardrail first: a reviewed answer citing a clause that was
never retrieved (or failing the PII check) is answered `422` with the reason and nothing is recorded -
fix it with `edit` and submit again. A passing answer is recorded, the graph checkpoint is resumed, and
the `200` response carries `released: true` and the final `answer`. Clause citations are not required
in a reviewer's text. `reject` records the decision and releases nothing. A second decision on the same
request is `409`.

### 17.8 `POST /ingest` — admin only, upload one policy document
Multipart file upload. Accepted types: `.md`, `.pdf`, `.docx`, `.txt`. `force=true` re-ingests even
if the file is unchanged.

```bash
curl -X POST "http://localhost:8000/ingest?force=false" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/Supplier_Vendor_Compliance_Policy.pdf"
```

### 17.9 `GET /ingest/status` — admin only
Corpus health: what is indexed, the active embed model and whether it is compatible, and
`unindexed_files`. This is the endpoint to check when `/ask` escalates with a no-match reason.

```bash
curl http://localhost:8000/ingest/status -H "Authorization: Bearer $TOKEN"
```

### 17.10 `POST /ingest/rebuild?confirm=true` — admin only
Empties the vector table and re-ingests every document in the corpus directory (one full embedding
pass). `confirm=true` is required or it refuses.

```bash
curl -X POST "http://localhost:8000/ingest/rebuild?confirm=true" \
  -H "Authorization: Bearer $TOKEN"
```

### 17.11 `GET /metrics/slo?hours=24` — reviewer only
Latency percentiles per stage and per evidence path, plus SLO breaches and outcome counts. `hours`
is 1–720 and defaults to the configured window.

```bash
curl "http://localhost:8000/metrics/slo?hours=24" -H "Authorization: Bearer $TOKEN"
```

### 17.11a `GET /metrics/optimization` — reviewer only
Cost and latency evidence (section 18): hit rates and the milliseconds / tokens each cache
returned (`caches`), how many routable model calls went to the small tier and why (`routing`),
and real token usage and USD spend per model and per tier with the per-request mean and peak
against `COST_BUDGET_USD_PER_REQUEST` (`cost`).

```bash
curl http://localhost:8000/metrics/optimization -H "Authorization: Bearer $TOKEN"
```

### 17.11b `POST /cache/invalidate?reason=...` — admin only
Drops every cached answer, chunk list and completion. The ingestion endpoints do this on their
own after the corpus changes; this is for a manual reset (a prompt edit in Langfuse, a seed
reload).

```bash
curl -X POST "http://localhost:8000/cache/invalidate?reason=prompt-change" -H "Authorization: Bearer $TOKEN"
```

### 17.12 Typical end-to-end flow

```
Auth0 login (frontend)    -> $TOKEN with the roles claim
GET  /auth/me             -> role, scopes, screens
POST /auth/azure          -> Azure OpenAI credentials from the login page
POST /ask                 -> 200 answered  (done)
                          -> 202 pending_review (has request_id + poll_url)
GET  /review/queue        -> reviewer sees it
GET  /review/{id}         -> reviewer reads the full package
POST /review/{id}         -> reviewer accepts / edits (200, released answer in the body)
                          -> 422 if the reviewed answer cannot be released (still in the queue)
                          -> or rejects (nothing released)
GET  /requests/{id}       -> caller polls: "answered" with the released answer, or "rejected"
```

### 17.13 Which role can call what

| Endpoint | Minimum role |
|---|---|
| `GET /health` | none |
| `GET /auth/me`, `POST /auth/azure`, `POST /ask`, `GET /requests/{id}` | any authenticated role |
| `GET /review/queue`, `GET /review/{id}`, `POST /review/{id}`, `GET /metrics/slo`, `GET /metrics/optimization` | compliance_officer / legal_reviewer / admin |
| `POST /ingest`, `GET /ingest/status`, `POST /ingest/rebuild`, `POST /cache/invalidate` | admin |

Note that `/ask` results also depend on role scope beyond authentication: record/SQL questions need
a role granted the relevant table (store_manager+ for `vendors`; compliance_officer+ for
`audit_logs` and `retention_records`), and GDPR/ISO documents are only readable by
compliance_officer and above.

---

## 18. Cost and latency: logging, caching and model routing

Three additions from the Course-4 Sprint-9 material ("optimise agentic systems for cost, latency
and quality") applied to this graph. Each one is measurable from the outside: the log shows what
happened per request, `GET /metrics/optimization` shows the totals, and the `/ask` trace shows the
decision for the request in hand.

### 18.1 Logging that follows a request

`src/observability/logging_config.py` stamps every log line with the id of the HTTP request being
served, `[req=<uuid>]`, from a context variable that the request-logging middleware in `main.py`
binds. `/ask` reuses the same id as the graph's `request_id`, so the access line, every node line,
the audit rows, the Langfuse trace and the `X-Request-ID` response header all agree. A client can
send its own `X-Request-ID` to correlate a retry. Threads started for the parallel legs of the
hybrid and panel paths inherit the id through `with_request_context(fn)`.

What gets logged, in order, for one answered question:

```
http_request_start method=POST path=/ask
ask START request_id=… user=… role=… use_cache=True
response cache MISS scope=…
input guardrail passed pii_redacted=none risk_floor=Medium risk_keywords=['retention period']
intent=policy_lookup entities=none document_scope=any
entity vendor='Acme' status=exact resolved_id=17         (one per entity)
risk fused=Medium scenario=none l1=Medium l2=Low l3_hits=0 disagreement=False
route intent=policy_lookup proposed=rag chosen=rag clamped=False
edge planner -> rag reason=budget headroom is sufficient
policy evidence chunks=6 scope=all best_score=0.81 elapsed_ms=412
model_route node=rag_generate tier=small deployment=gpt-4o-mini strategy=heuristic reason=…
llm_call node=rag_generate model=gpt-4o-mini elapsed_ms=1830 tokens[p/c/t]=2100/180/2280 usd=0.000423
validation PASSED tier=small deterministic=0 llm=0 blocking=none grounded_ratio=1.00
confidence OK final=0.9000 threshold=0.75 retrieval=0.800 validation=1.000 agreement=1.000 coverage=0.800
edge confidence_scoring -> respond
output guardrail RELEASED citation_coverage=1.00 citations=1 pii_removed=none
ask END outcome=answered path=rag total_ms=4210 llm_calls=5 llm_tokens=5310 usd=0.00091 routing=['rag_generate=small', 'compliance_validation=small']
response cache STORE request_id=… produced_in_ms=4210
http_request method=POST path=/ask status=200 elapsed_ms=4216
```

Refusals, clarifications, budget stops, degraded paths, guardrail rejections and escalations log
at WARNING with their reason; database, model and mail failures at ERROR with the traceback. The
noisy third-party loggers (httpx, presidio, llama_index, …) are pinned to WARNING. Startup prints
one line per operator-relevant setting (secrets are shown only as configured / not configured).
The old `configs/logger.py`, which used to install its own `basicConfig(force=True)` on import and
silently replace the rotating file handler, is now a shim over the central configuration.

### 18.2 Caching (`src/cache`)

Three caches at three depths, all process-local, bounded (LRU) and time-limited (TTL), all
reporting hits, misses and what they saved:

| cache | key | what a hit skips | invalidated when |
|---|---|---|---|
| **response** (`response_cache.py`) | access scope (role, scopes, departments, `document_scope`, as-of) + normalised question, or its embedding | the whole graph: every model call and every database read | corpus ingest / rebuild, `POST /cache/invalidate`, TTL (1 h) |
| **retrieval** (`retrieval_cache.py`) | question + doc types + scope + as-of + top-k/top-n + rerank flag | the embedding call, vector + BM25 search, fusion and the cross-encoder | same, TTL 10 min |
| **LLM** (`llm_cache.py`) | LangChain prompt + model configuration (deployment, temperature, bound schema) | one model call | same, TTL 1 h |

The response cache is the one that changes the numbers. It only ever stores a *released* answer:
`status == answered`, confidence at or above the threshold, not degraded, no budget stop, no
skipped optional node — so a cache hit is a certified answer being certified again, never a
short-cut past validation. Escalations, refusals and clarifications are never cached, and a
follow-up that carries conversation history is never looked up, because its meaning depends on
the thread. Two callers share an entry only when their access scope is identical; a store
associate cannot be served a compliance officer's answer.

A **semantic** hit embeds the new question with the same deployment the index uses and accepts
the nearest cached question in the same scope at cosine similarity ≥ `SEMANTIC_CACHE_THRESHOLD`
(0.95). "What is the retention period for invoices?" and "How long are invoices retained?" hit;
"Can we share customer data with an overseas vendor?" does not. With Azure unconfigured, or when
the embedding call fails, the cache degrades to exact match and says so in the log.

A hit is still a request: it gets a fresh `request_id`, an audit row (`response_cache /
cache_hit` naming the source request), a `request_latency` row with `evidence_path = cache` so
the SLO report shows the latency it returned, and both conversation turns are recorded so a
follow-up has history. Each entry remembers the wall-clock and token cost of the run that
produced it, which is how `saved_ms_total` and `saved_tokens_total` in the report are real numbers
rather than a hit rate.

The LLM cache is LangChain's global cache slot, filled with a TTL/LRU implementation instead of
the unbounded `InMemoryCache`. Only temperature-0 models use it; the panel challenger (0.4) and
the generation nodes (0.1) are built with `cache=False`. A hit shows in the log as
`llm_call … cached=true` and is booked in the cost ledger as saved rather than spent.

### 18.3 Model routing (`src/llm_routing`)

The Sprint-9 "Gateway Router" pattern: a zero-latency heuristic decides how much reasoning a call
needs, and the cheap tier serves everything that does not need the strong one. Nodes fall into
three classes:

- **fixed small** — rewrite, intent, risk classifier, planner, template selector, SQL narration,
  reflection. Structured-output calls the small model handles; unchanged.
- **fixed strong** — the four panel agents. A High-risk answer always gets the strongest model;
  the router never trades that for cost.
- **routable** — `rag_generate`, `hybrid_generate`, `compliance_validation`, `agentic_rag`. The
  calls that write or judge the answer, and the expensive ones. Here `MODEL_ROUTING_STRATEGY`
  decides:

| strategy | routable nodes use |
|---|---|
| `heuristic` (default) | `assess_complexity` → small for simple questions, strong for complex |
| `cost_saver` | small unless the fused risk is High |
| `quality_first` | strong, never downgraded |
| `static` | the fixed `TIER_FOR_NODE` table — the previous behaviour |

`assess_complexity` (`complexity.py`) is the reference router's keyword-and-length rule set
extended with what the graph already knows by the time an answer is generated: risk High →
complex; a complex trigger phrase (`cross-border`, `legal hold`, `override`, `conflict`, `gift`,
`hospitality`, `overdue`, `why`, …) → complex; more than `ROUTING_COMPLEX_WORD_COUNT` words →
complex; a multi-part question → complex; intent `compliance_check` / `incident_guidance` →
complex; evidence path hybrid / agentic / panel → complex; otherwise simple, with the simple
triggers (`what is the`, `retention period`, `how many`, `status of`, …) noted. The reasons are
kept, so the trace says *why* a question was judged complex, not just that it was.

**Latency pressure.** With less than `ROUTING_LATENCY_PRESSURE_SECONDS` of deadline or
`ROUTING_TOKEN_PRESSURE` tokens left, a routable node drops to the small tier so the request
finishes inside its SLO instead of escalating on a budget stop. High-risk requests are exempt
(correctness outranks speed there, and the panel path already carries the widest deadline), and
`quality_first` never downgrades.

Every decision is a `RoutingDecision` appended to the state's `model_routing` channel, so it is in
the `/ask` trace, the reviewer package and the log (`model_route node=… tier=… reason=…`).

**Cost ledger** (`ledger.py`, `pricing.py`). The `tokens_spent` channel is a budget *estimate* the
guard uses before a node runs; the ledger records what the provider actually billed, taken from
the token usage the logging callback sees on every call, priced from a per-million-token table
(`MODEL_PRICES_JSON` overrides it). Per request it goes into `trace.llm_usage`; process-wide into
`GET /metrics/optimization` as spend per model and per tier, the small-tier share, and the
per-request mean and peak against `COST_BUDGET_USD_PER_REQUEST` (a soft ceiling: exceeding it
logs a warning and is counted, never blocked). Calls made through llama-index (the agentic path,
the generated-SQL engine) bypass the LangChain callback and are not in the ledger.

**Gateway.** Set `LLM_GATEWAY_URL` to route both tiers through an OpenAI-compatible proxy —
`litellm_config.yaml` is the Sprint-9 layout with `simple-agent` / `complex-agent` mapped to the
two Azure deployments. The backend keeps making the same tier decisions; the gateway changes
where each tier is served from, so the small tier can be swapped for a local model without a code
change.

### 18.4 What to expect

For the monthly mix in the brief (30–40 % retention questions, 20–30 % vendor compliance, the rest
long-tail), the response cache absorbs the repeated questions and the heuristic router sends
the plain lookups — most retention and status questions — to the small tier. Both effects are
visible on `GET /metrics/optimization`: `caches.response.hit_rate` and `saved_ms_total`,
`routing.routable_small_share`, and `cost.by_tier`. Nothing on the high-risk path changes: High
risk still means the strong tier, the panel, and a human.

Tests: `tests/test_cache.py`, `tests/test_model_routing.py`, `tests/test_request_logging.py` and
`tests/test_graph_routing_integration.py` (the real graph and real nodes against a fake chat
model, proving a simple question is generated and validated on the small tier and a complex one
on the strong tier).

## 19. External MCP tool: DuckDuckGo search for regulation cross-checks

The agentic path already had an MCP client (`src/tools/mcp_tools.py`, section 1) but nothing to
talk to. It now talks to the open-source **DuckDuckGo MCP server**, which exposes two tools over
streamable HTTP: `search(query, max_results, region)` and `fetch_content(url, start_index,
max_length)`. The server itself is built on `httpx`; our side reaches it through
`llama_index.tools.mcp.BasicMCPClient`, which speaks the MCP JSON-RPC protocol over HTTP.

Why this tool and not a database or filesystem server: `policy_documents` and
`compliance_records` already cover everything inside the company. The one thing the agent could
not do was look *outside* — check that the internal privacy policy still matches the current text
of GDPR Art. 33, or confirm a regulator's notification window. A web tool fills that gap without
duplicating the guarded SQL layer.

### 19.1 How the pieces connect

```
duckduckgo-mcp-server (child process, stdio) ────────────────┐
                                                             │  MCP (JSON-RPC over stdio, or HTTP)
FastAPI /ask ─► LangGraph ─► agentic_rag_node ─► ReActAgent ─┤
                                     │                       │
                        build_tools_for(include_mcp=True)    │
                                     │                       │
                          McpToolProvider.load_tools() ◄─────┘
                          (allow-list, timeout, output clamp)
```

1. `McpToolProvider` reads `MCP_SERVER_URL`, opens an MCP session, calls `tools/list`, keeps only
   the names in `MCP_ALLOWED_TOOLS`, and wraps each one so its output is cut at
   `MCP_MAX_OUTPUT_CHARS` (6 000) — a fetched web page must not blow the token budget.
2. `registry.build_tools_for` appends those tools after the two internal ones, so the agent sees
   `[policy_documents, compliance_records, search, fetch_content]`.
3. The ReAct agent decides per step which tool to call. The system prompt in
   `src/nodes/agentic_rag.py` restricts the web tools to cross-checks *after* a policy lookup and
   forces web results to be labelled `[Web: <domain>]`, never presented as a policy clause.
4. `MCP_REQUIRED=false` keeps the API up when the server is down: the provider logs the failure,
   returns no tools, and the agent runs with the internal two.

### 19.2 Running it

`BasicMCPClient` accepts either a URL or a command in `MCP_SERVER_URL`. The default is the
**command form**: the backend spawns the DuckDuckGo server as a child process and talks to it
over stdio, so there is no second service to run, locally or on Render.

```bash
uv add duckduckgo-mcp-server          # once; lands in pyproject.toml

uv run python scripts/mcp_smoke.py "GDPR article 33 breach notification 72 hours"
#   [1/3] 'duckduckgo-mcp-server' is a command, not a URL - the client will spawn it over stdio
#   [2/3] MCP handshake ok, tools loaded: ['search', 'fetch_content']
#   [3/3] search(...) returned 2311 chars: ...

uv run uvicorn main:app --reload      # unchanged
```

`.env`:

```
MCP_ENABLED=true
MCP_REQUIRED=false
MCP_SERVER_URL=duckduckgo-mcp-server
MCP_ALLOWED_TOOLS=search,fetch_content
MCP_TIMEOUT_SECONDS=30
MCP_MAX_OUTPUT_CHARS=6000
```

Each agentic request spawns a fresh server process (about 0.5–1 s). If that ever matters, run
it once as a long-lived HTTP service instead and point the URL at it — nothing else changes:

```bash
uvx duckduckgo-mcp-server --transport streamable-http --port 7070
# .env: MCP_SERVER_URL=http://127.0.0.1:7070/mcp
```

On Render (native Python service, no Docker) the command form needs nothing beyond the
dependency: the build installs it, the start command stays `uvicorn main:app --host 0.0.0.0
--port $PORT`.

`tests/test_mcp_duckduckgo.py` is a live test: it pings the URL with `httpx`, skips when nothing
listens, and otherwise asserts the allow-list, the clamp and a real search result.

### 19.3 Seeing it work from Swagger (`http://localhost:8000/docs`)

Only the **agentic** evidence path uses MCP tools, so the request has to land there. The router
(section 15.2) sends `INCIDENT_GUIDANCE` intents to agentic by default and needs a role that can
read both documents and records, so use a compliance officer.

1. Log in through the frontend as a user with the `compliance_officer` role and copy the Auth0
   access token (browser devtools → Network → any `/ask` or `/auth/me` request → `Authorization`
   header). In Swagger click **Authorize** (top right), paste it, *Authorize*, *Close*.
2. **`POST /ask`** → *Try it out* → body
   ```json
   {
     "query": "We detected a personal data breach affecting customer records this morning. What does our privacy policy require us to do, and does that still match the current GDPR Article 33 deadline?",
     "use_cache": false
   }
   ```
   `use_cache: false` forces a graph run instead of a cached answer.
3. *Execute*. Expect a `200` after 10–40 s (the agentic deadline is 75 s). In the response body:
   - `evidence_path` is `"agentic"`.
   - `trace.path_decision` names the intent (`incident_guidance`) and why agentic was chosen.
   - `trace.reasoning_trace` / the node trace carries
     `"tools_available": ["policy_documents", "compliance_records", "search", "fetch_content"]`
     and `tool_calls` ≥ 2 — one internal lookup plus at least one web call.
   - `answer` cites policy clauses as `[Document Title §clause]` and the external check as
     `[Web: gdpr-info.eu]` (or whichever domain the search returned).
   - `citations` lists only the internal clauses; the web result is deliberately not a citation.
4. Confirm the call happened on the server side: the API log shows
   `agentic START ... tools=['policy_documents', 'compliance_records', 'search', 'fetch_content']`
   followed by the ReAct step that called `search`.
5. Negative test for the demo: set `MCP_SERVER_URL=` (empty) or to a command that does not
   exist, restart, repeat step 2. The API still returns `200`, `tools_available` shrinks to the
   two internal tools, and the log shows either `runs without external MCP tools` or
   `MCP tools could not be loaded` — that is the `MCP_REQUIRED=false` degradation.

A `202 pending_review` on step 3 is not an MCP failure: it means the risk classifier marked the
breach question high-risk and escalated it. Either lower the risk (ask "does our privacy policy
match GDPR Article 33's 72-hour deadline?") or approve it via **`GET /review/queue`** →
**`POST /review/{request_id}`** with `{"decision": "accept"}` and read the answer from
**`GET /requests/{request_id}`** (section 17.12).
