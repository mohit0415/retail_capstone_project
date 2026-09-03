from dataclasses import dataclass, field

from src.guardrails.injection import detect_injection
from src.guardrails.pii import redact
from src.guardrails.scope import is_in_domain, lexical_risk_floor


@dataclass(slots=True)
class GuardrailOutcome:
    blocked: bool
    sanitised_query: str
    refusal_reason: str = ""
    pii_entities: list[str] = field(default_factory=list)
    risk_floor: str = "Low"
    risk_keywords: list[str] = field(default_factory=list)


def run_input_guardrail(raw_query: str) -> GuardrailOutcome:
    injected, snippet = detect_injection(raw_query)

    if injected:
        return GuardrailOutcome(
            blocked=True,
            sanitised_query=raw_query,
            refusal_reason=f"request rejected by the prompt-injection filter (matched: {snippet})",
        )

    sanitised, pii_entities = redact(raw_query)

    in_domain, domain_reason = is_in_domain(sanitised)

    if not in_domain:
        return GuardrailOutcome(
            blocked=True,
            sanitised_query=sanitised,
            refusal_reason=domain_reason,
            pii_entities=pii_entities,
        )

    floor, keywords = lexical_risk_floor(sanitised)

    return GuardrailOutcome(
        blocked=False,
        sanitised_query=sanitised,
        pii_entities=pii_entities,
        risk_floor=floor,
        risk_keywords=keywords,
    )
