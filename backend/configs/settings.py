import logging
from datetime import date
from functools import lru_cache

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

MINIMUM_DEADLINES = {
    "deadline_seconds_standard": 30.0,
    "deadline_seconds_hybrid": 45.0,
    "deadline_seconds_agentic": 75.0,
    "deadline_seconds_high_risk": 90.0,
}

MINIMUM_TOKEN_BUDGET = 16000

VALID_ROUTING_STRATEGIES = frozenset({"static", "heuristic", "cost_saver", "quality_first"})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    log_level: str = "INFO"
    log_file: str = "logs/rpids.log"
    log_file_max_bytes: int = 10 * 1024 * 1024
    log_file_backup_count: int = 5

    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_small_deployment: str = "gpt-4o-mini"
    azure_openai_strong_deployment: str = "gpt-4o-mini"
    azure_openai_embedding_deployment: str = "text-embedding-3-small"
    azure_openai_vision_deployment: str = "gpt-4o-mini"
    embedding_dimensions: int = 1536

    llamaparse_api_key: str = ""
    vector_table_name: str = "policy_kb"
    images_storage_dir: str = "data/stored_images"

    policy_corpus_dir: str = "data/policies"
    bootstrap_corpus_on_startup: bool = True
    bootstrap_fail_fast: bool = False
    upload_max_bytes: int = 20 * 1024 * 1024

    max_chunk_tokens: int = 512
    max_chunk_chars: int = 2200
    chunk_overlap: int = 40
    chunk_overlap_chars: int = 200
    max_table_chars: int = 5000
    semantic_buffer_size: int = 1
    semantic_breakpoint_percentile: int = 95

    use_auto_retriever: bool = False
    fusion_num_queries: int = 1
    enable_flashrank_rerank: bool = True
    flashrank_model: str = "ms-marco-MiniLM-L-12-v2"
    flashrank_cache_dir: str = ""
    enable_citation_synthesis: bool = False

    # the v4 workflow (rag -> nl2sql -> multi-agent panel -> compliance validation) has no agentic
    # route and no intent admits one; kept so existing .env files still parse
    enable_agentic_path: bool = True
    agent_max_iterations: int = 8
    agentic_min_seconds: float = 3.0
    hybrid_min_seconds: float = 2.0

    mcp_enabled: bool = True
    mcp_required: bool = False
    mcp_server_url: str = ""
    mcp_allowed_tools: str = ""
    mcp_timeout_seconds: int = 30
    mcp_max_output_chars: int = 6000

    enable_guardrails_ai: bool = True
    # Presidio + spaCy en_core_web_lg need ~500 MB of RAM; small hosts turn this off
    # and PII redaction falls back to the regex patterns in src.guardrails.pii.
    enable_presidio_pii: bool = True
    pii_entities: str = ""

    escalation_email_enabled: bool = False
    escalation_email_min_risk: str = "High"
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    application_email: str = ""
    reviewer_email: str = ""
    reviewer_console_url: str = "http://localhost:8000/review"

    database_url: str = "postgresql://postgres:postgres@localhost:5432/retail_compliance_db"
    db_pool_min_size: int = 2
    db_pool_max_size: int = 10
    db_connect_timeout_seconds: float = 3.0
    sql_statement_timeout_ms: int = 3000
    sql_row_limit: int = 200
    enable_generated_sql_fallback: bool = True

    # Auth0 (same variable names as the practice-1 travel planner backend).
    # The frontend .env carries the matching VITE_AUTH0_* values.
    auth0_domain: str = ""
    api_audience: str = ""
    algorithms: str = "RS256"
    auth0_role_namespace: str = "https://stateful-agent.com"

    confidence_threshold: float = 0.75
    max_reflection_retries: int = 2
    panel_repair_passes: int = 1

    # ---- latency controls ----
    # one model call on a slow Azure deployment can hang; fail it after this many seconds and retry
    llm_timeout_seconds: float = 20.0
    llm_max_retries: int = 2
    # a repair pass (reflection -> plan -> evidence -> validation) needs at least this much deadline left
    repair_headroom_seconds: float = 8.0
    # a RAG repair re-asks the same documents, so one attempt is enough (hybrid/agentic keep MAX_REFLECTION_RETRIES)
    rag_repair_attempts: int = 1
    # a records repair keeps the rows and rewrites the answer once with the validator's defects
    sql_repair_attempts: int = 1
    # a drafted answer may still be validated this many seconds after the deadline instead of escalating
    validation_grace_seconds: float = 15.0
    # run the intent classifier and the risk classifier at the same time
    parallel_intent_and_risk: bool = True
    # do not ask the planner LLM when only one evidence path is possible anyway
    skip_planner_for_single_path: bool = True
    # do not ask the rewrite LLM when a follow-up already reads as a full question
    skip_rewrite_for_standalone: bool = True
    default_token_budget: int = 16000
    deadline_seconds_standard: float = 30.0
    deadline_seconds_hybrid: float = 45.0
    deadline_seconds_agentic: float = 75.0
    deadline_seconds_high_risk: float = 90.0

    conversation_window_turns: int = 10
    conversation_rewrite_turns: int = 5
    conversation_summary_after_turns: int = 8
    conversation_summary_input_turns: int = 30

    slo_t1_p95_ms: float = 6000.0
    slo_t2_p95_ms: float = 9000.0
    slo_t3_p95_ms: float = 20000.0
    slo_t4_p95_ms: float = 27000.0
    slo_total_p95_ms: float = 28000.0
    slo_path_target_ratio: float = 0.85
    slo_window_hours: int = 24

    # per-request RAGAS answer-quality scoring (background task after /ask)
    enable_ragas_eval: bool = True

    retrieval_top_k: int = 20
    rerank_top_n: int = 6
    rrf_k: int = 60
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # sibling-clause expansion: a top-ranked clause "6.1" pulls in its current
    # "6.x" siblings so a numbered section reaches the writer complete even when
    # the cross-encoder cuts the clauses whose meaning lives in their heading
    enable_sibling_expansion: bool = True
    sibling_expansion_seeds: int = 2
    max_sibling_clauses: int = 4

    as_of_date: date = date(2025, 12, 31)

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias=AliasChoices("LANGFUSE_HOST", "LANGFUSE_BASE_URL", "langfuse_host"),
    )
    tracing_enabled: bool = False

    rate_limit_standard: str = "30/minute"
    rate_limit_high_risk: str = "10/minute"

    enable_response_cache: bool = True
    response_cache_ttl_seconds: int = 3600
    response_cache_max_entries: int = 500
    enable_semantic_cache: bool = True
    semantic_cache_threshold: float = 0.95
    enable_llm_cache: bool = True
    llm_cache_ttl_seconds: int = 3600
    llm_cache_max_entries: int = 2000
    enable_retrieval_cache: bool = True
    retrieval_cache_ttl_seconds: int = 600
    retrieval_cache_max_entries: int = 500

    model_routing_strategy: str = "heuristic"
    routing_complex_word_count: int = 50
    routing_latency_pressure_seconds: float = 8.0
    routing_token_pressure: int = 6000
    llm_gateway_url: str = ""
    llm_gateway_api_key: str = "dummy-key"
    llm_gateway_small_model: str = "simple-agent"
    llm_gateway_strong_model: str = "complex-agent"
    use_model_routing_yaml: bool = False
    model_routing_yaml_path: str = "configs/model_routing.yaml"
    model_prices_json: str = ""
    cost_budget_usd_per_request: float = 0.05

    @model_validator(mode="after")
    def _raise_unworkable_budgets(self):
        raised = []

        for field, floor in MINIMUM_DEADLINES.items():
            configured = getattr(self, field)

            if configured < floor:
                object.__setattr__(self, field, floor)
                raised.append(f"{field.upper()} {configured} -> {floor}")

        if self.default_token_budget < MINIMUM_TOKEN_BUDGET:
            raised.append(f"DEFAULT_TOKEN_BUDGET {self.default_token_budget} -> {MINIMUM_TOKEN_BUDGET}")
            object.__setattr__(self, "default_token_budget", MINIMUM_TOKEN_BUDGET)

        if raised:
            logger.warning(
                "budget settings below the workable floor were raised (%s); a request makes seven "
                "or more model calls, so a smaller budget escalates every answer before it is scored. "
                "Fix these in .env rather than relying on this floor",
                ", ".join(raised),
            )

        strategy = (self.model_routing_strategy or "heuristic").strip().lower()

        if strategy not in VALID_ROUTING_STRATEGIES:
            logger.warning(
                "MODEL_ROUTING_STRATEGY=%r is not one of %s; using 'heuristic'",
                self.model_routing_strategy,
                sorted(VALID_ROUTING_STRATEGIES),
            )
            strategy = "heuristic"

        object.__setattr__(self, "model_routing_strategy", strategy)

        if self.use_model_routing_yaml and self.app_env.strip().lower() != "local":
            logger.warning(
                "USE_MODEL_ROUTING_YAML=true ignored because APP_ENV=%r is not 'local'; "
                "every tier is served by Azure OpenAI",
                self.app_env,
            )
            object.__setattr__(self, "use_model_routing_yaml", False)

        if not 0.0 < self.semantic_cache_threshold <= 1.0:
            logger.warning(
                "SEMANTIC_CACHE_THRESHOLD=%s is outside (0, 1]; using 0.95", self.semantic_cache_threshold
            )
            object.__setattr__(self, "semantic_cache_threshold", 0.95)

        return self

    @property
    def azure_configured(self) -> bool:
        return bool(self.azure_openai_endpoint and self.azure_openai_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
