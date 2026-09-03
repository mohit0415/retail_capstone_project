CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

DROP TABLE IF EXISTS policy_chunks;

CREATE TABLE IF NOT EXISTS vendors (
    vendor_id        SERIAL PRIMARY KEY,
    vendor_name      VARCHAR(100) NOT NULL,
    risk_score       INTEGER,
    risk_category    VARCHAR(20),
    compliance_status VARCHAR(30),
    approval_status  VARCHAR(30),
    onboarding_date  DATE,
    last_audit_date  DATE,
    next_review_due  DATE
);

CREATE TABLE IF NOT EXISTS audit_logs (
    audit_id              SERIAL PRIMARY KEY,
    vendor_id             INTEGER NOT NULL,
    policy_reference      VARCHAR(100),
    issue_title           VARCHAR(255),
    issue_severity        VARCHAR(20),
    remediation_status    VARCHAR(30),
    issue_identified_date DATE,
    target_resolution_date DATE,
    resolution_date       DATE,
    escalation_flag       BOOLEAN,

    CONSTRAINT fk_audit_vendor
        FOREIGN KEY (vendor_id)
        REFERENCES vendors (vendor_id)
);

CREATE TABLE IF NOT EXISTS retention_records (
    retention_id           SERIAL PRIMARY KEY,
    vendor_id              INTEGER NOT NULL,
    department             VARCHAR(50),
    data_category          VARCHAR(100),
    retention_period_years INTEGER,
    legal_hold_flag        BOOLEAN,
    approval_status        VARCHAR(30),
    last_review_date       DATE,
    next_review_due        DATE,

    CONSTRAINT fk_retention_vendor
        FOREIGN KEY (vendor_id)
        REFERENCES vendors (vendor_id)
);

CREATE TABLE IF NOT EXISTS compliance_reviews (
    review_id       SERIAL PRIMARY KEY,
    vendor_id       INTEGER NOT NULL,
    reviewer_name   VARCHAR(100),
    review_type     VARCHAR(50),
    review_status   VARCHAR(30),
    review_notes    TEXT,
    review_date     DATE,
    next_review_due DATE,

    CONSTRAINT fk_review_vendor
        FOREIGN KEY (vendor_id)
        REFERENCES vendors (vendor_id)
);

CREATE INDEX IF NOT EXISTS vendors_name_trgm_idx ON vendors USING GIN (lower(vendor_name) gin_trgm_ops);
CREATE INDEX IF NOT EXISTS vendors_risk_idx ON vendors (risk_category, compliance_status);
CREATE INDEX IF NOT EXISTS vendors_review_due_idx ON vendors (next_review_due);

CREATE INDEX IF NOT EXISTS audit_logs_vendor_idx ON audit_logs (vendor_id);
CREATE INDEX IF NOT EXISTS audit_logs_escalation_idx ON audit_logs (escalation_flag, remediation_status);
CREATE INDEX IF NOT EXISTS audit_logs_target_idx ON audit_logs (target_resolution_date);

CREATE INDEX IF NOT EXISTS retention_vendor_idx ON retention_records (vendor_id);
CREATE INDEX IF NOT EXISTS retention_department_idx ON retention_records (department);
CREATE INDEX IF NOT EXISTS retention_review_due_idx ON retention_records (next_review_due, legal_hold_flag);

CREATE INDEX IF NOT EXISTS compliance_reviews_vendor_idx ON compliance_reviews (vendor_id, review_date DESC);
CREATE INDEX IF NOT EXISTS compliance_reviews_status_idx ON compliance_reviews (review_status);

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

DROP RULE IF EXISTS system_audit_no_update ON system_audit_log;
DROP RULE IF EXISTS system_audit_no_delete ON system_audit_log;

CREATE RULE system_audit_no_update AS ON UPDATE TO system_audit_log DO INSTEAD NOTHING;
CREATE RULE system_audit_no_delete AS ON DELETE TO system_audit_log DO INSTEAD NOTHING;

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

DO $$
BEGIN
    IF to_regclass('public.data_policy_kb') IS NOT NULL THEN
        EXECUTE $view$
            CREATE OR REPLACE VIEW policy_kb_stats AS
            SELECT
                metadata_->>'doc_type'       AS doc_type,
                metadata_->>'document_title' AS document_title,
                metadata_->>'version'        AS version,
                metadata_->>'parsed_with'    AS parsed_with,
                metadata_->>'content_type'   AS content_type,
                COUNT(*)                     AS node_count
            FROM data_policy_kb
            GROUP BY 1, 2, 3, 4, 5
        $view$;
    END IF;
END
$$;
