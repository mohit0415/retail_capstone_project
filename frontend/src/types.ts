// src/types.ts
// These match the pydantic models in backend/src/schemas/api.py
// (I copied the field names from there so the UI stays in sync with the backend)

export type Role =
  | 'store_associate'
  | 'store_manager'
  | 'compliance_officer'
  | 'legal_reviewer'
  | 'admin'

export type Screen = 'chat' | 'dashboard' | 'review' | 'slo' | 'corpus'

// GET /auth/me
export interface MeResponse {
  user_id: string
  role: Role
  departments: string[]
  access_scopes: string[]
  auth0_roles: string[]
  email: string
  name: string
  permissions: {
    can_chat: boolean
    can_view_dashboard: boolean
    can_review: boolean
    can_view_slo: boolean
    can_ingest: boolean
  }
  screens: Screen[]
  azure_configured: boolean
}

// POST /auth/azure
export interface AzureCredentials {
  endpoint: string
  api_key: string
  api_version: string
  small_deployment: string
  strong_deployment: string
  embedding_deployment: string
  // the vector width follows the embedding model; the backend derives it when omitted
  embedding_dimensions?: number
  // LlamaCloud key for parsing PDFs with tables / diagrams. Typed here so the
  // parsing is billed to whoever is logged in, not to the server's own key.
  llamaparse_api_key: string
}

export interface AzureCredentialsResponse {
  azure_configured: boolean
  verified: boolean
  endpoint: string
  api_version: string
  small_deployment: string
  strong_deployment: string
  embedding_deployment: string
  embedding_dimensions: number
  llamaparse_configured: boolean
  llamaparse_source: 'login' | 'env' | 'none'
  warning: string
  message: string
}

// POST /ask
export interface AskRequest {
  query: string
  thread_id?: string | null
  document_scope?: string[]
  use_cache?: boolean
}

export interface Citation {
  document_title: string
  clause_number: string
  section: string
  version: string
  excerpt: string
}

export interface SqlProof {
  template_id: string
  statement: string
  parameters: Record<string, unknown>
  row_count: number
  as_of: string
}

export interface StageTimings {
  t1_ms?: number | null
  t2_ms?: number | null
  t3_ms?: number | null
  t4_ms?: number | null
  total_ms: number
}

export interface CacheInfo {
  hit: boolean
  kind: 'exact' | 'semantic'
  similarity: number
  age_seconds: number
  matched_query: string
  source_request_id: string
  saved_ms_estimate: number
  saved_tokens_estimate: number
}

// one step of the multi-agent workflow (trace.agent_steps from the backend)
export interface AgentStepChild {
  agent: string
  summary: string
}

export interface AgentStep {
  node: string
  agent: string
  kind: string // agent | guard | plan | reflect | step
  status: string // completed | budget_stopped
  elapsed_ms: number
  since_start_ms: number
  summary: string
  tier: string | null // small / strong, null when the step made no model call
  attempt: number
  children: AgentStepChild[]
}

export interface DecisionTrace {
  guardrail: string
  standalone_query: string
  intent: Record<string, unknown> | null
  resolved_entities: Record<string, unknown>[]
  risk: Record<string, unknown> | null
  plan: Record<string, unknown> | null
  routed_path: string | null
  path_decision: Record<string, unknown> | null
  evidence_path: string | null
  retrieved_chunks: string[]
  sql_evidence: SqlProof | null
  panel_verdict: Record<string, unknown> | null
  cited_clauses: string[]
  validation: Record<string, unknown> | null
  confidence: Record<string, unknown> | null
  tokens_spent: number
  token_budget: number
  deadline_seconds: number | null
  degraded: boolean
  skipped_optional_nodes: string[]
  budget_stops: Record<string, unknown>[]
  reflection_count: number
  panel_repair_count: number
  terminal_outcome: string | null
  escalation_reason: string | null
  escalation_reference: string | null
  model_routing: Record<string, unknown>[]
  llm_usage: Record<string, unknown> | null
  agent_steps?: AgentStep[] // older cached answers do not have it
}

export interface AnswerResponse {
  status: 'answered'
  request_id: string
  thread_id: string
  answer: string
  citations: Citation[]
  sql_evidence: SqlProof | null
  confidence: number
  risk_level: string
  uncertainty_note: string
  evidence_path: string
  degraded: boolean
  timings: StageTimings | null
  history_turns_used: number
  thread_summary_used: boolean
  standalone_query: string
  not_found?: boolean // true when the backend answered "I don't know" (nothing in the documents)
  not_found_reason?: string // no_match | not_in_extracts | no_records | no_evidence | no_access
  trace: DecisionTrace | null
  cache: CacheInfo | null
}

export interface PendingReviewResponse {
  status: 'pending_review'
  request_id: string
  thread_id: string
  risk_level: string
  reason: string
  escalation_reference: string | null
  queued_at: string
  poll_url: string
  timings: StageTimings | null
  trace: DecisionTrace | null
}

export interface RefusalResponse {
  status: 'refused'
  request_id: string
  reason: string
}

export interface ClarificationResponse {
  status: 'clarification_required'
  request_id: string
  thread_id: string
  question: string
  candidates: Record<string, unknown>[]
}

export type AskResponse = AnswerResponse | PendingReviewResponse | RefusalResponse | ClarificationResponse

// GET /requests/{request_id}
export interface RequestStatus {
  status: 'pending_review' | 'answered' | 'rejected'
  request_id: string
  thread_id: string
  risk_level: string
  reason?: string
  escalation_reference?: string | null
  queued_at?: string
  answer?: string
  reviewer_decision?: string
  reviewed_at?: string
  note?: string // set when status is 'rejected'
}

// review
export interface QueueItem {
  request_id: string
  thread_id: string
  user_id: string
  risk_level: string
  reason: string
  queued_at: string
}

export interface ReviewPackage {
  request_id: string
  thread_id: string
  original_query: string
  standalone_query: string
  conversation_history: Record<string, unknown>[]
  retrieved_documents: Record<string, unknown>[]
  sql_evidence: Record<string, unknown> | null
  validation_output: Record<string, unknown>
  reasoning_trace: Record<string, unknown>[]
  draft_answer: string
  risk_level: string
  confidence: number
  evidence_path: string | null
  panel_verdict: Record<string, unknown> | null
  plan: Record<string, unknown> | null
  panel_repair_count: number
  reflection_count: number
  reference_id: string | null
  budget_stops: Record<string, unknown>[]
  degraded: boolean
}

export type ReviewDecision = 'accept' | 'edit' | 'reject'

export interface ReviewOutcome {
  status: 'recorded'
  request_id: string
  decision: string
  resumed: boolean
  released: boolean
  outcome: string | null
  answer: string | null
  answer_source: string | null
  evidence_path: string | null
  note: string
}

// corpus
export interface IngestResponse {
  status: 'indexed' | 'skipped'
  file_name: string
  doc_type: string
  version: string
  parsed_with: string
  total_nodes: number
  text_nodes: number
  table_nodes: number
  image_nodes: number
  superseded_nodes: number
  reason: string
}

export interface CorpusStatusResponse {
  indexed: boolean
  corpus_dir: string
  embed_model: string
  embed_model_compatible: boolean
  message: string
  documents: Record<string, unknown>[]
  unindexed_files: string[]
}

export interface RebuildResponse {
  cleared_chunks: number
  corpus_dir: string
  embed_model: string
  indexed_files: string[]
  skipped_files: { file_name: string; reason: string }[]
  total_nodes: number
  documents: Record<string, unknown>[]
  unindexed_files: string[]
}

export interface CacheInvalidationResponse {
  reason: string
  cleared: Record<string, number>
}

// ops
export interface SloStage {
  stage: string
  meaning: string
  p50_ms: number | null
  p95_ms: number | null
  p99_ms: number | null
  target_p95_ms: number
  meets_slo: boolean | null
}

export interface SloReport {
  window_hours: number
  sample_size: number
  measured_from: string
  total_target_basis: string
  stages: SloStage[]
  total: {
    p50_ms: number | null
    p95_ms: number | null
    p99_ms: number | null
    target_p95_ms: number
    meets_slo: boolean | null
  }
  breached_stages: string[]
  breached_paths: string[]
  outcomes: { outcome: string; count: number }[]
  evidence_paths: {
    evidence_path: string
    count: number
    total_p95_ms: number | null
    target_p95_ms: number
    meets_slo: boolean | null
  }[]
}

export interface OptimizationReport {
  generated_at: string
  caches: Record<string, unknown>
  routing: Record<string, unknown>
  cost: Record<string, unknown>
}

export interface HealthResponse {
  status: string
  database: string
  audit_log: Record<string, unknown>
  escalation_queue_depth: number | null
  as_of_date: string
  confidence_threshold: number
  deadlines_seconds: Record<string, number>
  default_token_budget: number
  azure_configured: boolean
  optimization: Record<string, unknown>
}
