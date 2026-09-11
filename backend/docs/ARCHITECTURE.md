# Architecture — Retail Policy Intelligence & Decision Support System

Capstone deliverable 1. The six layers named in the brief, and where each one lives in this
repository. Renders on GitHub, in VS Code (Markdown Preview Mermaid) and in the Langfuse /
Swagger docs pages.

```mermaid
flowchart TB
    subgraph Clients["Callers"]
        UI["Compliance officer / store role<br/>Swagger · curl · front-end"]
        REV["Reviewer console<br/>(legal_reviewer / compliance_officer / admin)"]
    end

    subgraph API["1 · API layer — FastAPI (main.py, src/api)"]
        MW["RequestLoggingMiddleware<br/>X-Request-ID · access log · request context"]
        AUTH["JWT auth + RBAC<br/>src/auth (roles → doc/table scopes)"]
        RL["Rate limiter (slowapi)"]
        IDEM["Idempotency store"]
        RC["Response cache<br/>exact + semantic · scoped by role/dept/doc-scope<br/>src/cache/response_cache.py"]
        ASK["POST /ask · GET /requests/{id}"]
        REVIEW["GET/POST /review/*"]
        OPS["GET /health · /metrics/slo · /metrics/optimization<br/>POST /ingest · /ingest/rebuild · /cache/invalidate"]
    end

    subgraph ORCH["2 · Agent orchestration — LangGraph (src/graph, src/nodes)"]
        GRAPH["Policy graph · 18 nodes<br/>guardrail → rewrite → intent → entities → risk → planner<br/>→ evidence path → validation ⇄ reflection → confidence → output guardrail<br/>interrupt_after=escalation_manager"]
        ROUTER["Model router<br/>src/llm_routing · heuristic complexity → small / strong tier<br/>budget-pressure downgrade · cost ledger"]
        BUDGET["Budget guard<br/>per-path deadlines · token budget · optional-node degradation"]
    end

    subgraph KNOW["3 · Retrieval & knowledge layer"]
        ING["Ingestion pipeline<br/>src/ingestion · PyMuPDF / LlamaParse → clause, table, image nodes<br/>metadata stamping · versioning"]
        VEC[("PostgreSQL + pgvector<br/>policy_kb (text-embedding-3-small)")]
        BM25["BM25 leg"]
        FUSION["QueryFusionRetriever (RRF) → CurrentVersionFilter → FlashRank rerank<br/>src/retrieval"]
        RCACHE["Retrieval cache<br/>src/cache/retrieval_cache.py"]
        SQLP["Vetted SQL catalogue + generated NL2SQL fallback<br/>src/sqlpath (shape + scope guards)"]
        DB[("PostgreSQL<br/>vendors · audit_logs · retention_records · compliance_reviews")]
    end

    subgraph EXT["4 · External tools layer"]
        AZ["Azure OpenAI<br/>gpt-4o (strong) · gpt-4o-mini (small) · embeddings · vision"]
        GW["Optional LiteLLM gateway<br/>litellm_config.yaml · simple-agent / complex-agent"]
        LLMC["LLM completion cache<br/>src/cache/llm_cache.py"]
        MCP["MCP tools<br/>src/tools/mcp_tools.py (MCP_SERVER_URL)"]
        TOOLS["ReAct agent tools<br/>policy_documents · compliance_records"]
    end

    subgraph OBS["5 · Evaluation & observability layer"]
        LOG["Structured logs<br/>request-id stamped · logs/rpids.log (rotating)"]
        LF["Langfuse tracing + prompt registry<br/>src/observability/langfuse_callback.py · src/prompts"]
        SLO["SLO history<br/>request_latency table · t1–t4 stage marks · /metrics/slo"]
        COST["Cost & cache metrics<br/>/metrics/optimization"]
        EVAL["Golden-set evaluation<br/>evals/run_eval.py (53 cases)"]
    end

    subgraph HITL["6 · Human-in-the-loop layer"]
        ESC["Escalation manager<br/>context package → escalation_queue · e-mail (High risk)"]
        AUD[("system_audit_log<br/>append-only · circuit breaker")]
        QUEUE[("escalation_queue")]
        RESUME["Review decision → update_state → resume graph<br/>through the output guardrail"]
    end

    UI --> MW --> AUTH --> RL --> ASK
    ASK --> IDEM --> RC
    RC -- miss --> GRAPH
    RC -- hit --> UI
    REV --> REVIEW
    GRAPH --> ROUTER
    GRAPH --> BUDGET
    ROUTER --> LLMC --> AZ
    ROUTER -. LLM_GATEWAY_URL .-> GW --> AZ
    GRAPH --> FUSION
    FUSION --> RCACHE
    FUSION --> VEC
    FUSION --> BM25
    GRAPH --> SQLP --> DB
    GRAPH --> TOOLS --> MCP
    ING --> VEC
    OPS --> ING
    GRAPH --> ESC --> QUEUE
    GRAPH --> AUD
    ESC --> AUD
    REVIEW --> RESUME --> GRAPH
    GRAPH --> LOG
    GRAPH --> LF
    ASK --> SLO
    ASK --> COST
    EVAL --> ASK
```

## Request lifecycle in one line per layer

1. **API** — the middleware mints (or accepts) `X-Request-ID`, JWT → role → document/table
   scopes, the idempotency store replays a duplicate, the response cache returns a certified
   answer for the same question in the same scope (exact or semantic), otherwise the graph runs.
2. **Orchestration** — LangGraph runs the Plan–Reason–Act loop with the budget guard in front of
   every node and the model router choosing the tier for the generation and validation calls.
3. **Knowledge** — the RAG leg fuses vector and BM25 hits, filters superseded versions and reranks
   (cached per query + scope); the records leg selects a vetted SQL template, or falls back to
   guarded generated SQL, against the four compliance tables.
4. **External tools** — Azure OpenAI directly or through a LiteLLM gateway; the completion cache
   short-circuits identical prompts; MCP tools are available to the ReAct agent on the agentic
   path.
5. **Observability** — every line carries the request id; Langfuse holds the trace and the
   prompts; `request_latency` feeds the SLO report; the ledger feeds the cost report; the
   golden set measures TSR, high-risk misclassification and per-path P95.
6. **Human-in-the-loop** — High risk, low confidence, conflicts, budget stops and explicit
   requests escalate with the full context package; a reviewer's decision resumes the paused
   graph so the released answer still passes the output guardrail.

## Data stores

| store | tables | purpose |
|---|---|---|
| PostgreSQL + pgvector | `policy_kb` | clause / table-summary / image-caption nodes with version metadata |
| PostgreSQL | `vendors`, `audit_logs`, `retention_records`, `compliance_reviews` | structured compliance records (synthetic) |
| PostgreSQL | `system_audit_log`, `escalation_queue`, `request_latency`, `conversation_turns`, `conversation_threads` | audit trail, human review queue, SLO history, multi-turn memory |
| PostgreSQL (LangGraph) | checkpoint tables | graph state per thread, so a review can resume the run |
| process memory | response / retrieval / LLM caches, idempotency store, cost ledger | cost and latency optimisation (see README §18) |
