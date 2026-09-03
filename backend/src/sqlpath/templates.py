from configs.settings import settings

RISK_CATEGORIES = ["Low", "Medium", "High", "Critical"]
COMPLIANCE_STATUSES = ["Compliant", "Under Review", "Non-Compliant"]
APPROVAL_STATUSES = ["Approved", "Pending", "Rejected"]
ISSUE_SEVERITIES = ["Low", "Medium", "High", "Critical"]
REMEDIATION_STATUSES = ["Open", "In Progress", "Closed"]
REVIEW_STATUSES = ["Open", "In Progress", "Closed"]
REVIEW_TYPES = ["Quarterly Review", "Annual Certification", "Escalation Review"]
RETENTION_APPROVAL_STATUSES = ["Approved", "Pending"]
DEPARTMENTS = ["Finance", "Marketing", "HR", "IT", "Legal"]

SENIOR_RISK_CATEGORIES = ["High", "Critical"]

TABLE_DESCRIPTIONS = {
    "vendors": (
        "One row per third-party supplier. vendor_id SERIAL is the primary key and vendor_name is "
        "the display name. risk_score is an integer 40-95 and risk_category is the band derived "
        f"from it, one of {RISK_CATEGORIES}. compliance_status is one of {COMPLIANCE_STATUSES}. "
        f"approval_status is one of {APPROVAL_STATUSES}. Dates: onboarding_date, last_audit_date, "
        "next_review_due."
    ),
    "audit_logs": (
        "Compliance findings raised against a vendor, several rows per vendor. audit_id is the key "
        "and vendor_id is the foreign key to vendors. policy_reference names the policy the finding "
        "was raised under. issue_severity is one of "
        f"{ISSUE_SEVERITIES}. remediation_status is one of {REMEDIATION_STATUSES}. "
        "issue_identified_date is when it was raised, target_resolution_date is the deadline, "
        "resolution_date is NULL until it is fixed. escalation_flag is a stored boolean marking a "
        "finding that passed its deadline while still unresolved."
    ),
    "retention_records": (
        "Data retention obligations, one row per department and data category. retention_id is the "
        f"key and vendor_id points at the vendor holding the data. department is one of {DEPARTMENTS}. "
        "data_category names the class of data. retention_period_years is a duration in years, not "
        "an expiry date. legal_hold_flag is true when the data is preserved for a legal matter. "
        f"approval_status is one of {RETENTION_APPROVAL_STATUSES}. last_review_date and "
        "next_review_due track the review cycle."
    ),
    "compliance_reviews": (
        "Scheduled reviews of a vendor. review_id is the key and vendor_id is the foreign key. "
        f"reviewer_name is the person. review_type is one of {REVIEW_TYPES}. review_status is one "
        f"of {REVIEW_STATUSES}. review_notes is free text. review_date is when it happened and "
        "next_review_due is when the next one falls."
    ),
}

SCHEMA_NOTES = f"""Rules you must follow when writing SQL against this database:

- Produce a SELECT (or WITH ... SELECT) statement only. Never write, never modify.
- Never call CURRENT_DATE, NOW() or CURRENT_TIMESTAMP. Compare against the pinned as-of date
  '{settings.as_of_date}' written as a literal. Every figure this system reports is pinned to that
  date so answers stay reproducible and auditable.
- Always add LIMIT {settings.sql_row_limit} or lower.

Value sets are stored with capitals and spaces exactly as written below. A WHERE clause that
lowercases them or replaces a space with an underscore matches nothing and returns an empty result,
which reads as "no problems found" and is the most damaging mistake you can make here:
- vendors.risk_category: {RISK_CATEGORIES}
- vendors.compliance_status: {COMPLIANCE_STATUSES}
- vendors.approval_status: {APPROVAL_STATUSES}
- audit_logs.issue_severity: {ISSUE_SEVERITIES}
- audit_logs.remediation_status: {REMEDIATION_STATUSES}
- compliance_reviews.review_status: {REVIEW_STATUSES}
- compliance_reviews.review_type: {REVIEW_TYPES}
- retention_records.approval_status: {RETENTION_APPROVAL_STATUSES}
- retention_records.department: {DEPARTMENTS}
Use ILIKE or an exact literal from these lists. Never invent a value outside them.

Domain rules that the column types do not convey:
- compliance_status and approval_status are independent. A vendor can be Approved and
  Non-Compliant at the same time; that combination is exactly what a compliance question is usually
  asking about. Never substitute one column for the other.
- risk_category is a band over risk_score, not a judgement about the vendor's findings. A vendor in
  the Low band can still carry a Critical finding in audit_logs, so a question about severity must
  read audit_logs.issue_severity, not vendors.risk_category.
- A finding is open when remediation_status <> 'Closed'. resolution_date IS NULL means the same
  thing; prefer remediation_status.
- A finding is overdue when target_resolution_date < '{settings.as_of_date}' and remediation_status
  <> 'Closed'. escalation_flag is a stored value computed when the row was created, so it can
  disagree with that comparison. When a question asks what is overdue, compute it from the dates.
  When a question asks what was flagged for escalation, read escalation_flag.
- retention_period_years is a duration, not a deadline. There is no stored expiry date, so a
  retention obligation cannot be called expired from this table. The review cycle is what is
  trackable: a record is overdue for review when next_review_due < '{settings.as_of_date}'.
- legal_hold_flag = true means the data is held deliberately for a legal matter. Exclude those rows
  when counting retention problems unless the question asks about legal holds.
- A review is outstanding when review_status <> 'Closed'.
- Every table except vendors joins back to vendors on vendor_id. Join to vendors whenever the answer
  needs the vendor's name rather than its id.
- Return the row identifier (vendor_id, audit_id, retention_id or review_id) alongside the columns
  the question asks about, so the answer can cite which record it came from."""


def schema_notes_for(allowed_tables: set[str]) -> str:
    lines = [SCHEMA_NOTES, "", "Tables available to this role:"]

    for table in sorted(allowed_tables):
        description = TABLE_DESCRIPTIONS.get(table)

        if description:
            lines.append(f"- {table}: {description}")

    return "\n".join(lines)


def tables_visible_to(allowed_tables: set[str]) -> list[str]:
    return sorted(table for table in allowed_tables if table in TABLE_DESCRIPTIONS)
