import re
from dataclasses import dataclass, field

from src.guardrails.pii import redact

CITATION_PATTERN = re.compile(r"\[([^\]]+?§[^\]]+?)\]")

CLAIM_SPLIT = re.compile(r"(?<=[.!?])\s+")

NON_CLAIM_PREFIXES = (
    "in summary",
    "note that",
    "if you need",
    "please",
    "this answer",
    "i cannot",
    "the following",
)


@dataclass(slots=True)
class OutputGuardOutcome:
    answer: str
    citations_found: list[str] = field(default_factory=list)
    pii_removed: list[str] = field(default_factory=list)
    citation_coverage: float = 0.0
    enforcement_failed: bool = False
    failure_reason: str = ""


def extract_citations(answer: str) -> list[str]:
    return CITATION_PATTERN.findall(answer)


def _substantive_sentences(answer: str) -> list[str]:
    sentences = [part.strip() for part in CLAIM_SPLIT.split(answer) if len(part.strip()) > 25]

    return [s for s in sentences if not s.lower().startswith(NON_CLAIM_PREFIXES)]


def run_output_guardrail(answer: str, allowed_citations: list[str]) -> OutputGuardOutcome:
    from src.guardrails.guardrails_ai import get_scrubber

    passed, scrubbed, guard_findings = get_scrubber().scrub(answer)

    if not passed:
        return OutputGuardOutcome(
            answer=scrubbed,
            pii_removed=guard_findings,
            enforcement_failed=True,
            failure_reason="the PII safety check could not be completed, so the answer was withheld",
        )

    scrubbed, pii_removed = redact(scrubbed)
    pii_removed = sorted(set(pii_removed) | set(guard_findings))

    citations = extract_citations(scrubbed)

    unknown = [c for c in citations if c not in allowed_citations]

    if unknown:
        return OutputGuardOutcome(
            answer=scrubbed,
            citations_found=citations,
            pii_removed=pii_removed,
            enforcement_failed=True,
            failure_reason=f"answer cites clauses that were never retrieved: {unknown}",
        )

    sentences = _substantive_sentences(scrubbed)

    if not sentences:
        return OutputGuardOutcome(
            answer=scrubbed,
            citations_found=citations,
            pii_removed=pii_removed,
            citation_coverage=1.0 if citations else 0.0,
        )

    cited_sentences = [s for s in sentences if CITATION_PATTERN.search(s)]
    coverage = len(cited_sentences) / len(sentences)

    if not citations:
        return OutputGuardOutcome(
            answer=scrubbed,
            citations_found=citations,
            pii_removed=pii_removed,
            citation_coverage=0.0,
            enforcement_failed=True,
            failure_reason="answer makes substantive claims with no clause citation",
        )

    return OutputGuardOutcome(
        answer=scrubbed,
        citations_found=citations,
        pii_removed=pii_removed,
        citation_coverage=round(coverage, 3),
    )
