import logging
import re
import unicodedata
from dataclasses import dataclass, field

from src.guardrails.pii import redact

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[([^\]]+?§[^\]]+?)\]")

CITATION_NOISE = re.compile(r"[^0-9a-z§.]+")

CITATION_MARKER = re.compile(r"\s*§\s*")

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


def canonical_citation(citation: str) -> str:
    text = unicodedata.normalize("NFKC", citation or "").casefold()
    text = CITATION_NOISE.sub(" ", text)
    text = CITATION_MARKER.sub(" § ", text)
    text = re.sub(r"\s+", " ", text).strip()

    while text.endswith("."):
        text = text[:-1].strip()

    return text


def _substantive_sentences(answer: str) -> list[str]:
    sentences = [part.strip() for part in CLAIM_SPLIT.split(answer) if len(part.strip()) > 25]

    return [s for s in sentences if not s.lower().startswith(NON_CLAIM_PREFIXES)]


def run_output_guardrail(
    answer: str, allowed_citations: list[str], require_citations: bool = True
) -> OutputGuardOutcome:
    from src.guardrails.guardrails_ai import get_scrubber

    passed, scrubbed, guard_findings = get_scrubber().scrub(answer)

    if not passed:
        logger.error("output PII check could not complete; answer withheld findings=%s", guard_findings)

        return OutputGuardOutcome(
            answer=scrubbed,
            pii_removed=guard_findings,
            enforcement_failed=True,
            failure_reason="the PII safety check could not be completed, so the answer was withheld",
        )

    scrubbed, pii_removed = redact(scrubbed)
    pii_removed = sorted(set(pii_removed) | set(guard_findings))

    if pii_removed:
        logger.info("output PII redacted entities=%s", pii_removed)

    citations = extract_citations(scrubbed)

    allowed = {canonical_citation(entry) for entry in allowed_citations}
    unknown = [c for c in citations if canonical_citation(c) not in allowed]

    if unknown:
        logger.warning("output cites unretrieved clauses: %s (allowed=%d)", unknown, len(allowed))

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

    if not citations and not require_citations:
        # the "I don't know" answer makes no policy claim, so there is nothing to cite
        return OutputGuardOutcome(
            answer=scrubbed,
            citations_found=citations,
            pii_removed=pii_removed,
            citation_coverage=0.0,
        )

    if not citations:
        logger.warning("output has %d substantive sentence(s) and no clause citation", len(sentences))

        return OutputGuardOutcome(
            answer=scrubbed,
            citations_found=citations,
            pii_removed=pii_removed,
            citation_coverage=0.0,
            enforcement_failed=True,
            failure_reason="answer makes substantive claims with no clause citation",
        )

    logger.debug(
        "output guardrail coverage=%.2f cited_sentences=%d/%d citations=%d",
        coverage,
        len(cited_sentences),
        len(sentences),
        len(citations),
    )

    return OutputGuardOutcome(
        answer=scrubbed,
        citations_found=citations,
        pii_removed=pii_removed,
        citation_coverage=round(coverage, 3),
    )
