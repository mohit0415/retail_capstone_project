from src.schemas.enums import Role

# ---------------------------------------------------------------------------
# Auth0 role -> backend Role
#
# The role names you create in the Auth0 dashboard must be exactly the Role
# values: store_associate, store_manager, compliance_officer, legal_reviewer,
# admin. The Auth0 Action "add-roles-to-tokens" copies them into the access
# token, and src/auth/security.py reads them from there.
# ---------------------------------------------------------------------------

# most privileged first - if a user carries several roles this order decides
ROLE_PRIORITY = [
    Role.ADMIN,
    Role.LEGAL_REVIEWER,
    Role.COMPLIANCE_OFFICER,
    Role.STORE_MANAGER,
    Role.STORE_ASSOCIATE,
]

# the plain "user" role from the practice-1 tenant still gets the read-only chat
ROLE_ALIASES = {"user": Role.STORE_ASSOCIATE}


def resolve_role(auth0_roles: list[str]) -> Role | None:
    """Pick the backend Role for a list of Auth0 role names (None when nothing matches)."""
    found: set[Role] = set()

    for name in auth0_roles or []:
        key = str(name).strip().lower().replace("-", "_").replace(" ", "_")

        if key in ROLE_ALIASES:
            found.add(ROLE_ALIASES[key])
            continue

        try:
            found.add(Role(key))
        except ValueError:
            continue

    for role in ROLE_PRIORITY:
        if role in found:
            return role

    return None


# which frontend screens each role gets (Miro board "Role to screen routing")
SCREENS = {
    Role.STORE_ASSOCIATE: ["chat"],
    Role.STORE_MANAGER: ["chat", "dashboard"],
    Role.COMPLIANCE_OFFICER: ["chat", "dashboard", "review", "slo"],
    Role.LEGAL_REVIEWER: ["chat", "review", "slo"],
    Role.ADMIN: ["chat", "dashboard", "review", "slo", "corpus"],
}


def screens_for(role: Role) -> list[str]:
    return list(SCREENS[role])


DOCUMENT_SCOPES = {
    # the retention policy is plain policy text (how long each record type is kept), and
    # the chat suggests "What is the retention period for customer transaction records?"
    # to associates - without this grant that question could only ever escalate
    Role.STORE_ASSOCIATE: {"privacy_policy", "infosec_policy", "anti_bribery_policy", "retention_policy"},
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

DASHBOARD_ROLES = {Role.STORE_MANAGER, Role.COMPLIANCE_OFFICER, Role.ADMIN}


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


def can_view_dashboard(role: Role) -> bool:
    return role in DASHBOARD_ROLES


def permissions_for(role: Role) -> dict[str, bool]:
    """Flags the frontend uses to show / hide screens (the backend still enforces them)."""
    return {
        "can_chat": True,
        "can_view_dashboard": can_view_dashboard(role),
        "can_review": can_review(role),
        "can_view_slo": can_review(role),
        "can_ingest": can_ingest(role),
    }
