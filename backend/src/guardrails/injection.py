import re

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(the\s+)?(system|earlier)\s+prompt", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an|the)\s+", re.I),
    re.compile(r"reveal\s+(your\s+)?(system\s+prompt|instructions|rules)", re.I),
    re.compile(r"act\s+as\s+(if\s+)?(you\s+have\s+)?(no\s+)?(restrictions|guardrails)", re.I),
    re.compile(r"(print|output|dump)\s+(the\s+)?(env|environment|api[_\s]?key|secret)", re.I),
    re.compile(r"</?(system|assistant|instructions)>", re.I),
    re.compile(r"\bDROP\s+TABLE\b|\bDELETE\s+FROM\b|\bTRUNCATE\b|\bALTER\s+TABLE\b", re.I),
    re.compile(r";\s*--|\bUNION\s+SELECT\b", re.I),
]

RETRIEVED_TEXT_WRAPPER = (
    "<retrieved_document source=\"{source}\">\n"
    "{content}\n"
    "</retrieved_document>"
)


def detect_injection(text: str) -> tuple[bool, str]:
    for pattern in INJECTION_PATTERNS:
        match = pattern.search(text)

        if match:
            return True, match.group(0)[:80]

    return False, ""


def neutralise_retrieved(content: str, source: str) -> str:
    stripped = re.sub(r"</?(system|assistant|instructions|tool)>", "", content, flags=re.I)

    return RETRIEVED_TEXT_WRAPPER.format(source=source, content=stripped)
