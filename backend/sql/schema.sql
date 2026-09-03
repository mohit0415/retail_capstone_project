CREATE TABLE vendors (
    vendor_id SERIAL PRIMARY KEY,
    vendor_name VARCHAR(100) NOT NULL,
    risk_score INTEGER,
    risk_category VARCHAR(20),
    compliance_status VARCHAR(30),
    approval_status VARCHAR(30),
    onboarding_date DATE,
    last_audit_date DATE,
    next_review_due DATE
);


CREATE TABLE audit_logs (
    audit_id SERIAL PRIMARY KEY,

    vendor_id INTEGER NOT NULL,

    policy_reference VARCHAR(100),
    issue_title VARCHAR(255),
    issue_severity VARCHAR(20),
    remediation_status VARCHAR(30),

    issue_identified_date DATE,
    target_resolution_date DATE,
    resolution_date DATE,

    escalation_flag BOOLEAN,

    CONSTRAINT fk_audit_vendor
        FOREIGN KEY (vendor_id)
        REFERENCES vendors(vendor_id)
);


CREATE TABLE retention_records (
    retention_id SERIAL PRIMARY KEY,

    vendor_id INTEGER NOT NULL,

    department VARCHAR(50),
    data_category VARCHAR(100),
    retention_period_years INTEGER,
    legal_hold_flag BOOLEAN,
    approval_status VARCHAR(30),

    last_review_date DATE,
    next_review_due DATE,

    CONSTRAINT fk_retention_vendor
        FOREIGN KEY (vendor_id)
        REFERENCES vendors(vendor_id)
);

CREATE TABLE compliance_reviews (
    review_id SERIAL PRIMARY KEY,

    vendor_id INTEGER NOT NULL,

    reviewer_name VARCHAR(100),
    review_type VARCHAR(50),
    review_status VARCHAR(30),
    review_notes TEXT,

    review_date DATE,
    next_review_due DATE,

    CONSTRAINT fk_review_vendor
        FOREIGN KEY (vendor_id)
        REFERENCES vendors(vendor_id)
);