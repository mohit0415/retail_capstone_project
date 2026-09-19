

CREATE TABLE IF NOT EXISTS conversation_threads (
    thread_id   TEXT PRIMARY KEY,
    summary     TEXT,
    turn_count  INTEGER NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversation_turns (
    turn_id     BIGSERIAL PRIMARY KEY,
    thread_id   TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    request_id  TEXT NOT NULL,
    speaker     TEXT NOT NULL CHECK (speaker IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conversation_turns_thread
    ON conversation_turns (thread_id, turn_id DESC);

-- Append-only audit trail (src/core/audit.py)

CREATE TABLE IF NOT EXISTS system_audit_log (
    system_audit_id BIGSERIAL PRIMARY KEY,
    request_id      TEXT NOT NULL,
    thread_id       TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    role            TEXT NOT NULL,
    node            TEXT NOT NULL,
    event           TEXT NOT NULL,
    risk_level      TEXT,
    confidence      DOUBLE PRECISION,
    outcome         TEXT,
    escalated       BOOLEAN GENERATED ALWAYS AS (outcome = 'escalated') STORED,
    detail          JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS system_audit_request_idx ON system_audit_log (request_id);
CREATE INDEX IF NOT EXISTS system_audit_escalated_idx ON system_audit_log (escalated, created_at DESC);

-- Human review queue (src/nodes/escalation.py, src/api/escalation_service.py)

CREATE TABLE IF NOT EXISTS escalation_queue (
    request_id        TEXT PRIMARY KEY,
    thread_id         TEXT NOT NULL,
    user_id           TEXT NOT NULL,
    role              TEXT NOT NULL,
    risk_level        TEXT NOT NULL,
    reason            TEXT NOT NULL,
    context_package   JSONB NOT NULL,
    status            TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'reviewed', 'expired')),
    reviewer_id       TEXT,
    reviewer_decision TEXT,
    reviewed_answer   TEXT,
    reviewer_notes    TEXT,
    queued_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewed_at       TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS escalation_queue_status_idx ON escalation_queue (status, queued_at);

-- SLO latency history (src/observability/slo.py, GET /metrics/slo)

CREATE TABLE IF NOT EXISTS request_latency (
    latency_id    BIGSERIAL PRIMARY KEY,
    request_id    TEXT NOT NULL,
    thread_id     TEXT NOT NULL,
    role          TEXT NOT NULL,
    outcome       TEXT NOT NULL,
    evidence_path TEXT,
    risk_level    TEXT,
    degraded      BOOLEAN NOT NULL DEFAULT FALSE,
    t1_ms         DOUBLE PRECISION,
    t2_ms         DOUBLE PRECISION,
    t3_ms         DOUBLE PRECISION,
    t4_ms         DOUBLE PRECISION,
    total_ms      DOUBLE PRECISION NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS request_latency_created_idx ON request_latency (created_at DESC);
CREATE INDEX IF NOT EXISTS request_latency_outcome_idx ON request_latency (outcome, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS request_latency_request_idx ON request_latency (request_id);

-- Per-request RAGAS answer-quality scores (src/observability/ragas_eval.py,
-- GET /requests/{request_id}/evaluation, GET /metrics/ragas)

CREATE TABLE IF NOT EXISTS request_evaluations (
    evaluation_id     BIGSERIAL PRIMARY KEY,
    request_id        TEXT NOT NULL,
    thread_id         TEXT NOT NULL,
    evidence_path     TEXT,
    status            TEXT NOT NULL,            -- done | skipped | error
    skipped_reason    TEXT,
    faithfulness      DOUBLE PRECISION,
    answer_accuracy   DOUBLE PRECISION,
    context_precision DOUBLE PRECISION,
    context_recall    DOUBLE PRECISION,
    judge_model       TEXT,
    contexts_scored   INTEGER NOT NULL DEFAULT 0,
    duration_ms       DOUBLE PRECISION,
    error             TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS request_evaluations_request_idx ON request_evaluations (request_id);
CREATE INDEX IF NOT EXISTS request_evaluations_created_idx ON request_evaluations (created_at DESC);

-- databases created before the accuracy metric existed get the column on startup
ALTER TABLE request_evaluations ADD COLUMN IF NOT EXISTS answer_accuracy DOUBLE PRECISION;
