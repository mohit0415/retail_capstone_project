from dataclasses import dataclass
from typing import Literal

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

SCHEMA_NOTES = f"""Domain rules the column types do not convey:

- Every figure this system reports is pinned to the as-of date '{settings.as_of_date}'. Nothing reads
  the database clock, so answers stay reproducible and auditable.
- compliance_status and approval_status are independent. A vendor can be Approved and
  Non-Compliant at the same time; that combination is exactly what a compliance question is usually
  asking about. Never substitute one column for the other.
- risk_category is a band over risk_score, not a judgement about the vendor's findings. A vendor in
  the Low band can still carry a Critical finding in audit_logs, so a question about severity reads
  audit_logs.issue_severity, not vendors.risk_category.
- A finding is open when remediation_status <> 'Closed'.
- A finding is overdue when target_resolution_date is before the as-of date and it is still open.
  escalation_flag is a stored value computed when the row was created, so it can disagree with that
  comparison. Overdue is computed from the dates; flagged-for-escalation reads escalation_flag.
- retention_period_years is a duration, not a deadline. There is no stored expiry date, so a
  retention obligation cannot be called expired from this table. What is trackable is the review
  cycle: a record is overdue for review when next_review_due is before the as-of date.
- legal_hold_flag = true means the data is preserved deliberately for a legal matter. Those rows are
  excluded from retention problem counts unless the question is about legal holds.
- A review is outstanding when review_status <> 'Closed'."""


@dataclass(frozen=True)
class TemplateParameter:
    name: str
    kind: Literal["int", "string", "enum"]
    description: str
    allowed: tuple[str, ...] = ()


@dataclass(frozen=True)
class SqlTemplate:
    template_id: str
    answers: str
    tables: frozenset[str]
    parameters: tuple[TemplateParameter, ...]
    statement: str


VENDOR_ID = TemplateParameter(
    name="vendor_id",
    kind="int",
    description="numeric vendor_id resolved by entity resolution, never a name",
)

TEMPLATES: dict[str, SqlTemplate] = {
    "vendor_profile": SqlTemplate(
        template_id="vendor_profile",
        answers="the full compliance snapshot of one named vendor",
        tables=frozenset({"vendors"}),
        parameters=(VENDOR_ID,),
        statement="""
            SELECT vendor_id, vendor_name, risk_score, risk_category, compliance_status,
                   approval_status, onboarding_date, last_audit_date, next_review_due
            FROM vendors
            WHERE vendor_id = %(vendor_id)s
            LIMIT %(row_limit)s
        """,
    ),
    "vendor_search_by_name": SqlTemplate(
        template_id="vendor_search_by_name",
        answers="which vendors match a name fragment, with their status",
        tables=frozenset({"vendors"}),
        parameters=(
            TemplateParameter(
                name="name_fragment",
                kind="string",
                description="part of the vendor name as the user wrote it",
            ),
        ),
        statement="""
            SELECT vendor_id, vendor_name, risk_category, compliance_status, approval_status
            FROM vendors
            WHERE vendor_name ILIKE '%%' || %(name_fragment)s || '%%'
            ORDER BY vendor_name
            LIMIT %(row_limit)s
        """,
    ),
    "vendors_by_compliance_status": SqlTemplate(
        template_id="vendors_by_compliance_status",
        answers="which vendors sit in a given compliance status",
        tables=frozenset({"vendors"}),
        parameters=(
            TemplateParameter(
                name="compliance_status",
                kind="enum",
                description="the compliance status being asked about",
                allowed=tuple(COMPLIANCE_STATUSES),
            ),
        ),
        statement="""
            SELECT vendor_id, vendor_name, risk_category, compliance_status, approval_status,
                   last_audit_date, next_review_due
            FROM vendors
            WHERE compliance_status = %(compliance_status)s
            ORDER BY vendor_name
            LIMIT %(row_limit)s
        """,
    ),
    "vendors_by_approval_status": SqlTemplate(
        template_id="vendors_by_approval_status",
        answers="which vendors sit in a given approval status",
        tables=frozenset({"vendors"}),
        parameters=(
            TemplateParameter(
                name="approval_status",
                kind="enum",
                description="the approval status being asked about",
                allowed=tuple(APPROVAL_STATUSES),
            ),
        ),
        statement="""
            SELECT vendor_id, vendor_name, risk_category, compliance_status, approval_status,
                   onboarding_date
            FROM vendors
            WHERE approval_status = %(approval_status)s
            ORDER BY vendor_name
            LIMIT %(row_limit)s
        """,
    ),
    "vendors_by_risk_category": SqlTemplate(
        template_id="vendors_by_risk_category",
        answers="which vendors sit in a given risk band",
        tables=frozenset({"vendors"}),
        parameters=(
            TemplateParameter(
                name="risk_category",
                kind="enum",
                description="the risk band being asked about",
                allowed=tuple(RISK_CATEGORIES),
            ),
        ),
        statement="""
            SELECT vendor_id, vendor_name, risk_score, risk_category, compliance_status,
                   approval_status
            FROM vendors
            WHERE risk_category = %(risk_category)s
            ORDER BY risk_score DESC
            LIMIT %(row_limit)s
        """,
    ),
    "vendors_review_overdue": SqlTemplate(
        template_id="vendors_review_overdue",
        answers="which vendors are past their next review date as of the pinned date",
        tables=frozenset({"vendors"}),
        parameters=(),
        statement="""
            SELECT vendor_id, vendor_name, risk_category, compliance_status, approval_status,
                   last_audit_date, next_review_due
            FROM vendors
            WHERE next_review_due < %(as_of)s
            ORDER BY next_review_due
            LIMIT %(row_limit)s
        """,
    ),
    "vendor_open_findings": SqlTemplate(
        template_id="vendor_open_findings",
        answers="the open audit findings raised against one vendor",
        tables=frozenset({"audit_logs", "vendors"}),
        parameters=(VENDOR_ID,),
        statement="""
            SELECT a.audit_id, a.vendor_id, v.vendor_name, a.policy_reference, a.issue_title,
                   a.issue_severity, a.remediation_status, a.issue_identified_date,
                   a.target_resolution_date, a.escalation_flag
            FROM audit_logs a
            JOIN vendors v ON v.vendor_id = a.vendor_id
            WHERE a.vendor_id = %(vendor_id)s
              AND a.remediation_status <> 'Closed'
            ORDER BY a.target_resolution_date
            LIMIT %(row_limit)s
        """,
    ),
    "findings_by_severity": SqlTemplate(
        template_id="findings_by_severity",
        answers="every open finding at a given severity, across vendors",
        tables=frozenset({"audit_logs", "vendors"}),
        parameters=(
            TemplateParameter(
                name="issue_severity",
                kind="enum",
                description="the finding severity being asked about",
                allowed=tuple(ISSUE_SEVERITIES),
            ),
        ),
        statement="""
            SELECT a.audit_id, a.vendor_id, v.vendor_name, a.issue_title, a.issue_severity,
                   a.remediation_status, a.target_resolution_date
            FROM audit_logs a
            JOIN vendors v ON v.vendor_id = a.vendor_id
            WHERE a.issue_severity = %(issue_severity)s
              AND a.remediation_status <> 'Closed'
            ORDER BY a.target_resolution_date
            LIMIT %(row_limit)s
        """,
    ),
    "overdue_remediation": SqlTemplate(
        template_id="overdue_remediation",
        answers="which findings passed their remediation deadline and are still open",
        tables=frozenset({"audit_logs", "vendors"}),
        parameters=(),
        statement="""
            SELECT a.audit_id, a.vendor_id, v.vendor_name, a.issue_title, a.issue_severity,
                   a.remediation_status, a.target_resolution_date, a.escalation_flag
            FROM audit_logs a
            JOIN vendors v ON v.vendor_id = a.vendor_id
            WHERE a.remediation_status <> 'Closed'
              AND a.target_resolution_date < %(as_of)s
            ORDER BY a.target_resolution_date
            LIMIT %(row_limit)s
        """,
    ),
    "escalated_open_findings": SqlTemplate(
        template_id="escalated_open_findings",
        answers="which open findings carry the stored escalation flag",
        tables=frozenset({"audit_logs", "vendors"}),
        parameters=(),
        statement="""
            SELECT a.audit_id, a.vendor_id, v.vendor_name, a.issue_title, a.issue_severity,
                   a.remediation_status, a.target_resolution_date, a.escalation_flag
            FROM audit_logs a
            JOIN vendors v ON v.vendor_id = a.vendor_id
            WHERE a.escalation_flag = TRUE
              AND a.remediation_status <> 'Closed'
            ORDER BY a.issue_severity DESC, a.target_resolution_date
            LIMIT %(row_limit)s
        """,
    ),
    "retention_overdue_review": SqlTemplate(
        template_id="retention_overdue_review",
        answers="which retention records are overdue for review, excluding legal holds",
        tables=frozenset({"retention_records", "vendors"}),
        parameters=(),
        statement="""
            SELECT r.retention_id, r.vendor_id, v.vendor_name, r.department, r.data_category,
                   r.retention_period_years, r.approval_status, r.last_review_date, r.next_review_due
            FROM retention_records r
            JOIN vendors v ON v.vendor_id = r.vendor_id
            WHERE r.next_review_due < %(as_of)s
              AND r.legal_hold_flag = FALSE
            ORDER BY r.next_review_due
            LIMIT %(row_limit)s
        """,
    ),
    "retention_by_department": SqlTemplate(
        template_id="retention_by_department",
        answers="the retention obligations held by one department",
        tables=frozenset({"retention_records", "vendors"}),
        parameters=(
            TemplateParameter(
                name="department",
                kind="enum",
                description="the owning department",
                allowed=tuple(DEPARTMENTS),
            ),
        ),
        statement="""
            SELECT r.retention_id, r.vendor_id, v.vendor_name, r.department, r.data_category,
                   r.retention_period_years, r.legal_hold_flag, r.approval_status, r.next_review_due
            FROM retention_records r
            JOIN vendors v ON v.vendor_id = r.vendor_id
            WHERE r.department = %(department)s
            ORDER BY r.next_review_due
            LIMIT %(row_limit)s
        """,
    ),
    "retention_legal_holds": SqlTemplate(
        template_id="retention_legal_holds",
        answers="which retention records are under a legal hold",
        tables=frozenset({"retention_records", "vendors"}),
        parameters=(),
        statement="""
            SELECT r.retention_id, r.vendor_id, v.vendor_name, r.department, r.data_category,
                   r.legal_hold_flag, r.approval_status, r.next_review_due
            FROM retention_records r
            JOIN vendors v ON v.vendor_id = r.vendor_id
            WHERE r.legal_hold_flag = TRUE
            ORDER BY v.vendor_name
            LIMIT %(row_limit)s
        """,
    ),
    "vendor_retention_records": SqlTemplate(
        template_id="vendor_retention_records",
        answers="the retention records held against one vendor",
        tables=frozenset({"retention_records", "vendors"}),
        parameters=(VENDOR_ID,),
        statement="""
            SELECT r.retention_id, r.vendor_id, v.vendor_name, r.department, r.data_category,
                   r.retention_period_years, r.legal_hold_flag, r.approval_status,
                   r.last_review_date, r.next_review_due
            FROM retention_records r
            JOIN vendors v ON v.vendor_id = r.vendor_id
            WHERE r.vendor_id = %(vendor_id)s
            ORDER BY r.next_review_due
            LIMIT %(row_limit)s
        """,
    ),
    "vendor_review_history": SqlTemplate(
        template_id="vendor_review_history",
        answers="the review history recorded against one vendor",
        tables=frozenset({"compliance_reviews", "vendors"}),
        parameters=(VENDOR_ID,),
        statement="""
            SELECT c.review_id, c.vendor_id, v.vendor_name, c.reviewer_name, c.review_type,
                   c.review_status, c.review_date, c.next_review_due
            FROM compliance_reviews c
            JOIN vendors v ON v.vendor_id = c.vendor_id
            WHERE c.vendor_id = %(vendor_id)s
            ORDER BY c.review_date DESC
            LIMIT %(row_limit)s
        """,
    ),
    "open_reviews_by_type": SqlTemplate(
        template_id="open_reviews_by_type",
        answers="which reviews of a given type are still outstanding",
        tables=frozenset({"compliance_reviews", "vendors"}),
        parameters=(
            TemplateParameter(
                name="review_type",
                kind="enum",
                description="the kind of review being asked about",
                allowed=tuple(REVIEW_TYPES),
            ),
        ),
        statement="""
            SELECT c.review_id, c.vendor_id, v.vendor_name, c.reviewer_name, c.review_type,
                   c.review_status, c.review_date, c.next_review_due
            FROM compliance_reviews c
            JOIN vendors v ON v.vendor_id = c.vendor_id
            WHERE c.review_type = %(review_type)s
              AND c.review_status <> 'Closed'
            ORDER BY c.next_review_due
            LIMIT %(row_limit)s
        """,
    ),
    "reviews_due": SqlTemplate(
        template_id="reviews_due",
        answers="which reviews fall due on or before the pinned as-of date",
        tables=frozenset({"compliance_reviews", "vendors"}),
        parameters=(),
        statement="""
            SELECT c.review_id, c.vendor_id, v.vendor_name, c.review_type, c.review_status,
                   c.review_date, c.next_review_due
            FROM compliance_reviews c
            JOIN vendors v ON v.vendor_id = c.vendor_id
            WHERE c.next_review_due <= %(as_of)s
              AND c.review_status <> 'Closed'
            ORDER BY c.next_review_due
            LIMIT %(row_limit)s
        """,
    ),
    "all_vendors": SqlTemplate(
        template_id="all_vendors",
        answers="the full vendor roster with each vendor's status, when no filter is named",
        tables=frozenset({"vendors"}),
        parameters=(),
        statement="""
            SELECT vendor_id, vendor_name, risk_score, risk_category, compliance_status,
                   approval_status
            FROM vendors
            ORDER BY risk_score DESC
            LIMIT %(row_limit)s
        """,
    ),
    "vendor_count_by_risk": SqlTemplate(
        template_id="vendor_count_by_risk",
        answers="how many vendors fall into each risk category (a count per band)",
        tables=frozenset({"vendors"}),
        parameters=(),
        statement="""
            SELECT risk_category, COUNT(*) AS vendor_count
            FROM vendors
            GROUP BY risk_category
            ORDER BY vendor_count DESC
            LIMIT %(row_limit)s
        """,
    ),
    "vendor_count_by_compliance": SqlTemplate(
        template_id="vendor_count_by_compliance",
        answers="how many vendors sit in each compliance status (a count per status)",
        tables=frozenset({"vendors"}),
        parameters=(),
        statement="""
            SELECT compliance_status, COUNT(*) AS vendor_count
            FROM vendors
            GROUP BY compliance_status
            ORDER BY vendor_count DESC
            LIMIT %(row_limit)s
        """,
    ),
    "open_findings_count_by_vendor": SqlTemplate(
        template_id="open_findings_count_by_vendor",
        answers=(
            "how many open findings each vendor carries, split out by severe and overdue - "
            "the count-per-vendor question, including 'count critical findings per vendor'"
        ),
        tables=frozenset({"audit_logs", "vendors"}),
        parameters=(),
        statement="""
            SELECT v.vendor_id, v.vendor_name, v.risk_category, v.compliance_status,
                   COUNT(*) AS open_findings,
                   COUNT(*) FILTER (WHERE a.issue_severity IN ('High', 'Critical'))
                       AS severe_open_findings,
                   COUNT(*) FILTER (WHERE a.target_resolution_date < %(as_of)s)
                       AS overdue_findings
            FROM audit_logs a
            JOIN vendors v ON v.vendor_id = a.vendor_id
            WHERE a.remediation_status <> 'Closed'
            GROUP BY v.vendor_id, v.vendor_name, v.risk_category, v.compliance_status
            ORDER BY severe_open_findings DESC, open_findings DESC
            LIMIT %(row_limit)s
        """,
    ),
    "senior_risk_vendors_with_open_findings": SqlTemplate(
        template_id="senior_risk_vendors_with_open_findings",
        answers=(
            "which High or Critical band vendors still carry an unresolved severe finding - "
            "whether senior-risk vendors are aligned with remediation timelines"
        ),
        tables=frozenset({"audit_logs", "vendors"}),
        parameters=(),
        statement="""
            SELECT DISTINCT v.vendor_id, v.vendor_name, v.risk_score, v.risk_category,
                   v.compliance_status, v.approval_status, a.issue_severity,
                   a.remediation_status, a.target_resolution_date
            FROM vendors v
            JOIN audit_logs a ON a.vendor_id = v.vendor_id
            WHERE v.risk_category IN ('High', 'Critical')
              AND a.remediation_status <> 'Closed'
              AND a.issue_severity IN ('High', 'Critical')
            ORDER BY v.risk_score DESC, a.target_resolution_date
            LIMIT %(row_limit)s
        """,
    ),
    "vendors_under_escalation": SqlTemplate(
        template_id="vendors_under_escalation",
        answers=(
            "which vendors have BOTH an outstanding review and an unresolved severe finding - "
            "the vendors genuinely in an escalation state"
        ),
        tables=frozenset({"audit_logs", "compliance_reviews", "vendors"}),
        parameters=(),
        statement="""
            SELECT DISTINCT v.vendor_id, v.vendor_name, v.risk_score, v.risk_category,
                   v.compliance_status, c.review_type, c.review_status, c.next_review_due
            FROM vendors v
            JOIN compliance_reviews c ON c.vendor_id = v.vendor_id
            JOIN audit_logs a ON a.vendor_id = v.vendor_id
            WHERE c.review_status <> 'Closed'
              AND a.remediation_status <> 'Closed'
              AND a.issue_severity IN ('High', 'Critical')
            ORDER BY v.risk_score DESC
            LIMIT %(row_limit)s
        """,
    ),
}

RESERVED_PARAMETERS = {"as_of", "row_limit"}


class TemplateBindingError(ValueError):
    pass


def tables_visible_to(allowed_tables: set[str]) -> list[str]:
    return sorted(table for table in allowed_tables if table in TABLE_DESCRIPTIONS)


def templates_visible_to(allowed_tables: set[str]) -> list[SqlTemplate]:
    granted = set(tables_visible_to(allowed_tables))

    return [template for template in TEMPLATES.values() if template.tables <= granted]


def get_template(template_id: str) -> SqlTemplate | None:
    return TEMPLATES.get(template_id)


def schema_notes_for(allowed_tables: set[str]) -> str:
    lines = [SCHEMA_NOTES, "", "Tables this role may read:"]

    for table in tables_visible_to(allowed_tables):
        description = TABLE_DESCRIPTIONS.get(table)

        if description:
            lines.append(f"- {table}: {description}")

    return "\n".join(lines)


def catalogue_for(templates: list[SqlTemplate]) -> str:
    lines = []

    for template in templates:
        lines.append(f"- {template.template_id}: {template.answers}")

        if not template.parameters:
            lines.append("    parameters: none")
            continue

        for parameter in template.parameters:
            detail = f"    parameter {parameter.name} ({parameter.kind}): {parameter.description}"

            if parameter.allowed:
                detail += f". One of {list(parameter.allowed)}"

            lines.append(detail)

    return "\n".join(lines)


def _coerce(parameter: TemplateParameter, raw):
    if raw is None:
        raise TemplateBindingError(f"parameter '{parameter.name}' is required but was not supplied")

    if parameter.kind == "int":
        try:
            return int(str(raw).strip())
        except (TypeError, ValueError) as exc:
            raise TemplateBindingError(
                f"parameter '{parameter.name}' must be a whole number, got {raw!r}"
            ) from exc

    value = str(raw).strip()

    if not value:
        raise TemplateBindingError(f"parameter '{parameter.name}' was empty")

    if parameter.kind == "enum":
        for candidate in parameter.allowed:
            if candidate.casefold() == value.casefold():
                return candidate

        raise TemplateBindingError(
            f"parameter '{parameter.name}' must be one of {list(parameter.allowed)}, got {value!r}"
        )

    return value


def bind_parameters(template: SqlTemplate, supplied: dict, as_of, row_limit: int | None = None) -> dict:
    bound = {
        "as_of": as_of,
        "row_limit": row_limit or settings.sql_row_limit,
    }

    supplied = supplied or {}

    for parameter in template.parameters:
        bound[parameter.name] = _coerce(parameter, supplied.get(parameter.name))

    unexpected = set(supplied) - {p.name for p in template.parameters} - RESERVED_PARAMETERS

    if unexpected:
        raise TemplateBindingError(
            f"template '{template.template_id}' does not accept {sorted(unexpected)}"
        )

    return bound


def flatten(statement: str) -> str:
    return " ".join(statement.split())
