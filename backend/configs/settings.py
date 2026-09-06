import logging
from datetime import date
from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

MINIMUM_DEADLINES = {
    "deadline_seconds_standard": 30.0,
    "deadline_seconds_hybrid": 45.0,
    "deadline_seconds_agentic": 75.0,
    "deadline_seconds_high_risk": 90.0,
}

MINIMUM_TOKEN_BUDGET = 16000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    log_level: str = "INFO"

    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_small_deployment: str = "gpt-4o-mini"
    azure_openai_strong_deployment: str = "gpt-4o"
    azure_openai_embedding_deployment: str = "text-embedding-3-small"
    azure_openai_vision_deployment: str = "gpt-4o"
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
    pii_entities: str = ""

    escalation_email_enabled: bool = False
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

    jwt_secret: str = "change-this-to-a-long-random-string"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60

    confidence_threshold: float = 0.75
    max_reflection_retries: int = 2
    panel_repair_passes: int = 1
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

    retrieval_top_k: int = 20
    rerank_top_n: int = 6
    rrf_k: int = 60
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    as_of_date: date = date(2025, 12, 31)

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    tracing_enabled: bool = False

    rate_limit_standard: str = "30/minute"
    rate_limit_high_risk: str = "10/minute"

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

        return self

    @property
    def azure_configured(self) -> bool:
        return bool(self.azure_openai_endpoint and self.azure_openai_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
