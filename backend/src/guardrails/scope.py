import re

IN_DOMAIN_TERMS = {
    "policy",
    "policies",
    "clause",
    "compliance",
    "gdpr",
    "iso",
    "retention",
    "vendor",
    "supplier",
    "privacy",
    "consent",
    "breach",
    "incident",
    "audit",
    "bribery",
    "gift",
    "hospitality",
    "infosec",
    "security",
    "data",
    "record",
    "records",
    "erasure",
    "subject",
    "dpo",
    "contract",
    "renewal",
    "review",
    "approval",
    "certification",
    "store",
    "employee",
    "customer",
    "pci",
    "encryption",
    "access",
    "disposal",
    "archive",
    "compliant",
    "regulation",
    "regulatory",
    "risk",
    "sanction",
    "remediation",
    "escalation",
    "onboarding",
    "diligence",
    "personal",
    "pii",
    "archival",
    "certificate",
    "training",
    "third",
}

PLURAL_SUFFIXES = ("ies", "es", "s")

OUT_OF_DOMAIN_MARKERS = [
    re.compile(r"\b(write|compose)\s+(me\s+)?(a\s+)?(poem|song|story|joke)\b", re.I),
    re.compile(r"\b(weather|football|cricket|recipe|movie|stock price|horoscope)\b", re.I),
    re.compile(r"\bwho\s+(won|is the president|is the ceo of)\b", re.I),
]

RISK_KEYWORD_FLOORS = {
    "High": [
        "data breach",
        "personal data breach",
        "regulatory notification",
        "supervisory authority",
        "sanction",
        "penalty",
        "litigation",
        "criminal",
        "bribe",
        "kickback",
        "whistleblower",
        "terminate contract",
        "delete customer data",
        "erasure request",
        "right to be forgotten",
    ],
    "Medium": [
        "expired",
        "overdue",
        "non-compliant",
        "noncompliant",
        "escalate",
        "exception",
        "waiver",
        "audit finding",
        "remediation",
        "conflict of interest",
        "vendor risk",
        "retention period",
        "dispose",
    ],
}


def singular(token: str) -> str:
    if token in IN_DOMAIN_TERMS:
        return token

    for suffix in PLURAL_SUFFIXES:
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            stem = token[: -len(suffix)]

            if suffix == "ies":
                stem = f"{stem}y"

            if stem in IN_DOMAIN_TERMS:
                return stem

    return token


def is_in_domain(text: str) -> tuple[bool, str]:
    lowered = text.lower()

    for marker in OUT_OF_DOMAIN_MARKERS:
        if marker.search(lowered):
            return False, "request is outside the retail policy and compliance domain"

    tokens = {singular(token) for token in re.findall(r"[a-z0-9']+", lowered)}

    if tokens & IN_DOMAIN_TERMS:
        return True, ""

    return False, "no policy or compliance subject could be identified in the request"


def lexical_risk_floor(text: str) -> tuple[str, list[str]]:
    lowered = text.lower()
    matched: list[str] = []

    for keyword in RISK_KEYWORD_FLOORS["High"]:
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            matched.append(keyword)

    if matched:
        return "High", matched

    for keyword in RISK_KEYWORD_FLOORS["Medium"]:
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            matched.append(keyword)

    if matched:
        return "Medium", matched

    return "Low", []
