# Retail Policy Intelligence & Decision Support System  
## Standardized Core Dataset Package

This document defines the mandatory dataset and baseline specifications for implementing the Retail Enterprise Policy Intelligence & Decision Support System capstone.

All teams must use this standardized dataset to ensure fairness, benchmarking consistency, and SLO comparability.

---

# 📦 PART 1 — The Starter Pack

## 📄 Policy Documents (Unstructured – RAG)

### 1. Retail Data Protection & Customer Privacy Policy
Covers:
- Customer PII handling
- Consent management
- Data sharing restrictions
- Data breach obligations
- Cross-border data transfer rules

---

### 2. Data Retention & Archival Policy
Covers:
- Retention schedules by data category
- Deletion workflows
- Legal hold exceptions
- Archival requirements

---

### 3. Supplier & Vendor Compliance Policy
Covers:
- Vendor onboarding due diligence
- Risk classification model
- Ongoing compliance checks
- Contractual obligations

---

### 4. Anti-Bribery & Ethical Conduct Policy
Covers:
- Gifts & hospitality rules
- Facilitation payments
- Conflict of interest
- Reporting and whistleblower protections

---

### 5. Information Security & Access Control Policy
Covers:
- Access provisioning
- Role-based access control
- MFA requirements
- Privileged account audits
- Incident response triggers

---

## 📘 Regulatory & Framework Excerpts

### 6. GDPR Selected Articles (Excerpt)
Include:
- Article 5
- Article 6
- Article 17
- Article 32

---

### 7. ISO 27001 Access Control Summary
Include:
- Control objectives
- Access review cadence
- Logging requirements

---

## 📊 Structured Compliance Data (SQL-backed)

### 8. Vendor Compliance Registry

### 9. Audit Findings & Remediation Log

### 10. Data Retention Approval Records

---

# 🗄 PART 2 — SQL Schema & Sample Synthetic Data

## Database: retail_compliance_db

---

## Table 1: vendors

```sql
CREATE TABLE vendors (
    vendor_id SERIAL PRIMARY KEY,
    vendor_name VARCHAR(150) NOT NULL,
    risk_score INTEGER CHECK (risk_score BETWEEN 0 AND 100),
    risk_category VARCHAR(50),
    compliance_status VARCHAR(50),
    approval_status VARCHAR(50),
    onboarding_date DATE,
    last_audit_date DATE,
    next_review_due DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

```

### Sample Data

```sql
INSERT INTO vendors 
(vendor_id, vendor_name, risk_score, risk_category, compliance_status, approval_status, onboarding_date, last_audit_date, next_review_due)
VALUES
(1, 'ABC Logistics', 82, 'High', 'Under Review', 'Pending', '2023-03-15', '2025-01-10', '2025-07-10'),
(2, 'Global Supplies Ltd', 65, 'Medium', 'Compliant', 'Approved', '2022-06-01', '2024-12-15', '2025-12-15'),
(3, 'RetailTech Services', 45, 'Low', 'Compliant', 'Approved', '2024-01-10', '2025-02-01', '2026-02-01'),
(4, 'Prime Distribution Co', 91, 'Critical', 'Non-Compliant', 'Rejected', '2021-11-20', '2024-11-30', '2025-05-30'),
(5, 'Nova Warehousing', 73, 'High', 'Under Review', 'Pending', '2023-09-05', '2025-01-20', '2025-08-20');

```

## Table 2: audit_logs

```sql
CREATE TABLE audit_logs (
    audit_id SERIAL PRIMARY KEY,
    vendor_id INTEGER REFERENCES vendors(vendor_id) ON DELETE CASCADE,
    policy_reference VARCHAR(150),
    issue_title VARCHAR(200),
    issue_severity VARCHAR(50),
    remediation_status VARCHAR(50),
    issue_identified_date DATE,
    target_resolution_date DATE,
    resolution_date DATE,
    escalation_flag BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

```

### Sample Data

```sql
INSERT INTO audit_logs
(audit_id, vendor_id, policy_reference, issue_title, issue_severity, remediation_status,
 issue_identified_date, target_resolution_date, resolution_date, escalation_flag)
VALUES
(1, 1, 'Vendor Compliance Policy', 'Incomplete Due Diligence Documentation', 'High',
 'Open', '2025-01-05', '2025-02-15', NULL, TRUE),

(2, 2, 'Data Retention Policy', 'Retention Schedule Not Updated', 'Medium',
 'Closed', '2024-11-01', '2025-01-01', '2024-12-20', FALSE),

(3, 4, 'Access Control Policy', 'Excess Privileged Access Detected', 'Critical',
 'In Progress', '2025-01-10', '2025-02-10', NULL, TRUE),

(4, 5, 'Anti-Bribery Policy', 'Incomplete Conflict Disclosure', 'High',
 'Open', '2025-01-12', '2025-02-20', NULL, FALSE);
```

## Table 3: retentions_records

```sql
CREATE TABLE retention_records (
    retention_id SERIAL PRIMARY KEY,
    department VARCHAR(100),
    vendor_id INTEGER REFERENCES vendors(vendor_id) ON DELETE SET NULL,
    data_category VARCHAR(150),
    retention_period_years INTEGER,
    legal_hold_flag BOOLEAN,
    approval_status VARCHAR(50),
    last_review_date DATE,
    next_review_due DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

```
### Sample Data

```sql
INSERT INTO retention_records
(retention_id, department, vendor_id, data_category, retention_period_years,
 legal_hold_flag, approval_status, last_review_date, next_review_due)
VALUES
(1, 'Finance', 1, 'Transaction Records', 7, TRUE, 'Approved', '2024-12-01', '2025-12-01'),
(2, 'Marketing', 2, 'Customer Email Data', 3, FALSE, 'Approved', '2025-01-01', '2026-01-01'),
(3, 'HR', 4, 'Employee Records', 5, FALSE, 'Pending', '2024-10-01', '2025-10-01'),
(4, 'IT', 3, 'Security Logs', 2, FALSE, 'Approved', '2025-01-15', '2026-01-15');

```
---

## Table 4: compliance_reviews

```sql
CREATE TABLE compliance_reviews (
    review_id SERIAL PRIMARY KEY,
    vendor_id INTEGER REFERENCES vendors(vendor_id) ON DELETE CASCADE,
    reviewer_name VARCHAR(150),
    review_type VARCHAR(100),
    review_status VARCHAR(50),
    review_notes TEXT,
    review_date DATE,
    next_review_due DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```
### Sample Data

```sql
INSERT INTO compliance_reviews
(review_id, vendor_id, reviewer_name, review_type, review_status,
 review_notes, review_date, next_review_due)
VALUES
(1, 1, 'Anita Sharma', 'Quarterly Compliance Review', 'Open',
 'Pending high-risk issue resolution', '2025-01-15', '2025-04-15'),

(2, 2, 'Rahul Mehta', 'Annual Compliance Certification', 'Closed',
 'No material findings', '2024-12-10', '2025-12-10'),

(3, 4, 'Priya Iyer', 'Critical Risk Escalation Review', 'In Progress',
 'Awaiting remediation plan', '2025-01-18', '2025-03-18');
```
---

### Note: 

**The starter dataset validates correctness.
For performance testing, risk modeling, and hybrid query evaluation, you must generate scaled synthetic data using the [provided script](./generate_capstone_sql_data.py).**


# 📊 PART 3 — Golden Query Distribution Template (50 Queries)

All teams must create and label 50 queries following this structure:

## Distribution

| Category | Count | Type |
|----------|-------|------|
| Policy Interpretation | 15 | RAG |
| Structured Lookup | 10 | SQL |
| Hybrid Reasoning | 10 | RAG + SQL |
| High-Risk Regulatory | 10 | Multi-Agent + Guardrail |
| Escalation Scenarios | 5 | Human Handoff |

## Example Query Types

### Policy Interpretation (RAG)
* "What are the retention rules for customer email data?"
* "What triggers vendor escalation under our compliance policy?"

### Structured Lookup (SQL)
* "List vendors with open high-severity audit findings."
* "Show vendors whose next_review_due is within 30 days."
* "Identify vendors with overdue remediation actions."
* "Count critical findings per vendor."

### Hybrid
* "Do vendors under legal hold comply with retention requirements?"
* "Are critical-risk vendors aligned with audit remediation timelines?"
* "Does our policy require suspension for vendors with active escalation flags?"

### High-Risk
* "Does GDPR Article 32 require suspension of vendors with unresolved data security findings?"
* "Is offering hospitality to suppliers under compliance investigation allowed?"
* "Should we continue operations with a vendor marked Non-Compliant and Critical risk?"

### Escalation
* "Override vendor rejection despite open critical findings."
* "Approve retention deletion while legal_hold_flag = TRUE."
* "Close review for vendor with active escalation flag."

## 📌 Mandatory Rules

* All teams must use this dataset.
* Teams may extend but not replace the core dataset.
* Golden queries must be labeled with:
   * Query Type
   * Risk Level
   * Expected Retrieval Mode
   * Expected Escalation (Yes/No)

