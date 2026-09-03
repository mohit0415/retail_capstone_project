from src.schemas.enums import Role

DOCUMENT_SCOPES = {
    Role.STORE_ASSOCIATE: {"privacy_policy", "infosec_policy", "anti_bribery_policy"},
    Role.STORE_MANAGER: {
        "privacy_policy",
        "infosec_policy",
        "anti_bribery_policy",
        "vendor_policy",
        "retention_policy",
    },
    Role.COMPLIANCE_OFFICER: {
        "privacy_policy",
        "infosec_policy",
        "anti_bribery_policy",
        "vendor_policy",
        "retention_policy",
        "gdpr",
        "iso_27001",
    },
    Role.LEGAL_REVIEWER: {
        "privacy_policy",
        "infosec_policy",
        "anti_bribery_policy",
        "vendor_policy",
        "retention_policy",
        "gdpr",
        "iso_27001",
    },
    Role.ADMIN: {
        "privacy_policy",
        "infosec_policy",
        "anti_bribery_policy",
        "vendor_policy",
        "retention_policy",
        "gdpr",
        "iso_27001",
    },
}

TABLE_SCOPES = {
    Role.STORE_ASSOCIATE: set(),
    Role.STORE_MANAGER: {"vendors", "compliance_reviews"},
    Role.COMPLIANCE_OFFICER: {"vendors", "compliance_reviews", "audit_logs", "retention_records"},
    Role.LEGAL_REVIEWER: {"vendors", "compliance_reviews", "audit_logs", "retention_records"},
    Role.ADMIN: {"vendors", "compliance_reviews", "audit_logs", "retention_records"},
}

RISK_CATEGORY_SCOPES = {
    Role.STORE_ASSOCIATE: set(),
    Role.STORE_MANAGER: {"Low", "Medium"},
    Role.COMPLIANCE_OFFICER: {"Low", "Medium", "High", "Critical"},
    Role.LEGAL_REVIEWER: {"Low", "Medium", "High", "Critical"},
    Role.ADMIN: {"Low", "Medium", "High", "Critical"},
}

DEPARTMENT_RESTRICTED_ROLES = {Role.STORE_ASSOCIATE, Role.STORE_MANAGER}

DEPARTMENT_SCOPED_TABLES = {"retention_records"}

RISK_SCOPED_TABLES = {"vendors"}

REVIEWER_ROLES = {Role.COMPLIANCE_OFFICER, Role.LEGAL_REVIEWER, Role.ADMIN}

CORPUS_ADMIN_ROLES = {Role.ADMIN}


def access_scopes_for(role: Role) -> list[str]:
    docs = sorted(f"doc:{name}" for name in DOCUMENT_SCOPES[role])
    tables = sorted(f"table:{name}" for name in TABLE_SCOPES[role])
    risk = sorted(f"risk:{name}" for name in RISK_CATEGORY_SCOPES[role])

    return docs + tables + risk


def allowed_doc_types(scopes: list[str]) -> list[str]:
    return [scope.removeprefix("doc:") for scope in scopes if scope.startswith("doc:")]


def allowed_tables(scopes: list[str]) -> set[str]:
    return {scope.removeprefix("table:") for scope in scopes if scope.startswith("table:")}


def allowed_risk_categories(scopes: list[str]) -> set[str]:
    return {scope.removeprefix("risk:") for scope in scopes if scope.startswith("risk:")}


def is_department_restricted(role: Role) -> bool:
    return role in DEPARTMENT_RESTRICTED_ROLES


def can_review(role: Role) -> bool:
    return role in REVIEWER_ROLES


def can_ingest(role: Role) -> bool:
    return role in CORPUS_ADMIN_ROLES
