"""Heuristic complexity assessment (no LLM, zero latency).

This is the Sprint-9 ``assess_complexity`` idea adapted to compliance
questions. The reference router looked only at the ticket text; here the
graph already knows more by the time an answer is generated - the fused risk
level, the classified intent, the evidence path - and each of those is a
stronger signal than any keyword, so they are consulted first.

Rules, in order (the first that fires decides):

1. Risk level High                          -> complex (the panel is mandatory anyway)
2. A complex trigger phrase is present      -> complex
3. More words than ROUTING_COMPLEX_WORD_COUNT -> complex
4. Multi-part question (two questions, "and also", "as well as", "both ... and") -> complex
5. Intent is compliance_check / incident_guidance -> complex
6. Evidence path is hybrid / agentic / high_risk_panel -> complex
7. A simple trigger phrase is present       -> simple
8. Default                                  -> simple

A score in [0, 1] is reported alongside the label so the trace shows *how*
complex the router thought the question was, not just which side of the
line it fell on.
"""

import re
from dataclasses import dataclass, field

from configs.settings import settings

COMPLEX_KEYWORDS = (
    "cross-border",
    "cross border",
    "jurisdiction",
    "legal hold",
    "override",
    "overrid",
    "conflict",
    "contradict",
    "versus",
    " vs ",
    "compare",
    "comparison",
    "difference between",
    "exception",
    "breach",
    "incident",
    "penalt",
    "fine ",
    "sanction",
    "escalat",
    "gift",
    "hospitality",
    "brib",
    "kickback",
    "transfer",
    "overseas",
    "international",
    "critical",
    "audit finding",
    "overdue",
    "precedence",
    "supersede",
    "justify",
    "implication",
    "consequence",
    "why ",
    "trade-off",
    "risk assessment",
    "root cause",
    "remediat",
)

SIMPLE_KEYWORDS = (
    "what is the",
    "what are the",
    "define",
    "definition",
    "retention period",
    "how long",
    "who is",
    "who owns",
    "status of",
    "list ",
    "which vendors",
    "how many",
    "when was",
    "when is",
    "is it allowed",
    "does the policy",
    "what does",
    "summarise",
    "summarize",
    "approval status",
    "next review",
)

COMPLEX_INTENTS = frozenset({"compliance_check", "incident_guidance"})

COMPLEX_PATHS = frozenset({"hybrid", "agentic", "high_risk_panel"})

_MULTI_PART = re.compile(r"\b(and also|as well as|in addition|additionally|both .+ and)\b", re.I)


@dataclass(slots=True)
class ComplexityAssessment:
    label: str
    tier: str
    score: float
    reasons: list[str] = field(default_factory=list)
    word_count: int = 0
    matched_complex: list[str] = field(default_factory=list)
    matched_simple: list[str] = field(default_factory=list)

    @property
    def complex(self) -> bool:
        return self.label == "complex"

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "tier": self.tier,
            "score": self.score,
            "reasons": list(self.reasons),
            "word_count": self.word_count,
            "matched_complex": list(self.matched_complex),
            "matched_simple": list(self.matched_simple),
        }


def _matches(text: str, keywords: tuple[str, ...]) -> list[str]:
    padded = f" {text} "

    return [keyword.strip() for keyword in keywords if keyword in padded]


def assess_complexity(
    query: str,
    *,
    risk_level: str | None = None,
    intent: str | None = None,
    evidence_path: str | None = None,
    entity_count: int = 0,
    history_turns: int = 0,
    word_threshold: int | None = None,
) -> ComplexityAssessment:
    """Return a :class:`ComplexityAssessment` for ``query`` and its graph context."""
    text = " ".join((query or "").split()).lower()
    words = text.split()
    word_count = len(words)
    threshold = word_threshold or settings.routing_complex_word_count

    matched_complex = _matches(text, COMPLEX_KEYWORDS)
    matched_simple = _matches(text, SIMPLE_KEYWORDS)
    question_marks = text.count("?")
    multi_part = question_marks > 1 or bool(_MULTI_PART.search(text))

    reasons: list[str] = []
    score = 0.0

    if (risk_level or "").lower() == "high":
        reasons.append("risk level is High")
        score += 0.6
    elif (risk_level or "").lower() == "medium":
        reasons.append("risk level is Medium")
        score += 0.25

    if matched_complex:
        reasons.append(f"complex triggers: {matched_complex[:4]}")
        score += min(0.5, 0.2 * len(matched_complex))

    if word_count > threshold:
        reasons.append(f"{word_count} words exceeds {threshold}")
        score += 0.3

    if multi_part:
        reasons.append("multi-part question")
        score += 0.2

    if intent in COMPLEX_INTENTS:
        reasons.append(f"intent {intent} needs rule-against-record reasoning")
        score += 0.3

    if evidence_path in COMPLEX_PATHS:
        reasons.append(f"evidence path {evidence_path} joins two sources")
        score += 0.3

    if entity_count > 1:
        reasons.append(f"{entity_count} entities to reconcile")
        score += 0.1

    if history_turns > 0:
        reasons.append(f"{history_turns} prior turns of context")
        score += 0.05

    complex_by_rule = bool(
        (risk_level or "").lower() == "high"
        or matched_complex
        or word_count > threshold
        or multi_part
        or intent in COMPLEX_INTENTS
        or evidence_path in COMPLEX_PATHS
    )

    if complex_by_rule:
        label = "complex"
    else:
        label = "simple"

        if matched_simple:
            reasons.append(f"simple triggers: {matched_simple[:4]}")
            score = max(0.0, score - 0.1)
        else:
            reasons.append("no complexity trigger matched")

    return ComplexityAssessment(
        label=label,
        tier="strong" if label == "complex" else "small",
        score=round(min(1.0, score), 3),
        reasons=reasons,
        word_count=word_count,
        matched_complex=matched_complex,
        matched_simple=matched_simple,
    )
