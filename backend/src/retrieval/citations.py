"""Which clause identifiers an answer may cite from the extracts it was given.

Every extract is cited as "<document title> §<clause_number>". The numbered headings inside an
extract's text are citable too: a section "4 Customer PII Handling" whose text carries
"4.1 Data Minimization" may be cited as §4.1.

This matters most for plain-text PDFs indexed before their headings were recognised: the whole
privacy policy became one "General §1" section. The model cites the clause numbers it reads in
the text (§4.1, §5.2) and the validator and the output guardrail used to reject every one of them
as "never retrieved" - 7 of 7 citations in the log - so a correct answer escalated.
"""

import re
from dataclasses import dataclass

from src.guardrails.output_guard import (
    CITATION_PATTERN,
    NON_CLAIM_PREFIXES,
    _substantive_sentences,
    canonical_citation,
)

UNNUMBERED_SECTION = "general"

UNNUMBERED_CLAUSE = "1"

MAX_HEADING_WORDS = 9

SUB_CLAUSE_HEADING = re.compile(
    r"^[ \t]*(?:[#>*-][ \t#>*-]*)?(?:(?:§|Section|Clause|Article)[ \t]*)?"
    r"([0-9]{1,2}(?:\.[0-9]{1,3})*|[A-Z](?:\.[0-9]{1,3})+)\.?[ \t]+"
    r"(?:\*\*)?([A-Z][^\n]*?)(?:\*\*)?[ \t]*$",
    re.M,
)


@dataclass(frozen=True)
class CitableClause:
    citation: str
    chunk: object
    clause_number: str
    heading: str
    offset: int = 0


def _clean_title(title: str) -> str:
    return title.strip().strip("*").strip()


def _looks_like_heading(title: str) -> bool:
    title = _clean_title(title)

    return bool(title) and not title.endswith((".", ":", ";", ",")) and len(title.split()) <= MAX_HEADING_WORDS


def is_whole_document_section(chunk) -> bool:
    return (getattr(chunk, "clause_number", "") or "").strip() == UNNUMBERED_CLAUSE and (
        getattr(chunk, "section", "") or ""
    ).strip().lower() == UNNUMBERED_SECTION


def sub_clause_headings(chunk) -> list[tuple[str, str, int]]:
    """(number, heading, offset) for each numbered heading inside the extract that may be cited.

    Inside a numbered section only its own sub-clauses count (§4 may be cited as §4.1, not as
    §7), so a numbered list in the text is not mistaken for a clause. A whole-document "General
    §1" section has no real number of its own, so every heading in it counts.
    """
    own = (getattr(chunk, "clause_number", "") or "").strip()
    whole = is_whole_document_section(chunk)
    found: list[tuple[str, str, int]] = []
    seen: set[str] = set()

    for match in SUB_CLAUSE_HEADING.finditer(getattr(chunk, "content", "") or ""):
        number, title = match.group(1), match.group(2)

        if number == own or number in seen or not _looks_like_heading(title):
            continue

        if not whole and not number.startswith(f"{own}."):
            continue

        seen.add(number)
        found.append((number, _clean_title(title), match.start()))

    return found


def citation_keys(chunk) -> set[str]:
    """Every canonical citation that points at this extract: its own and its inner headings'."""
    keys = {canonical_citation(chunk.citation)}
    keys.update(
        canonical_citation(f"{chunk.document_title} §{number}") for number, _heading, _offset in sub_clause_headings(chunk)
    )

    return keys


def citable_clauses(chunks) -> dict[str, CitableClause]:
    """canonical citation -> the clause it names, for every extract and every heading inside one."""
    table: dict[str, CitableClause] = {}

    for chunk in chunks or []:
        table.setdefault(
            canonical_citation(chunk.citation),
            CitableClause(
                citation=chunk.citation,
                chunk=chunk,
                clause_number=chunk.clause_number,
                heading=chunk.section,
            ),
        )

    for chunk in chunks or []:
        for number, heading, offset in sub_clause_headings(chunk):
            citation = f"{chunk.document_title} §{number}"
            table.setdefault(
                canonical_citation(citation),
                CitableClause(citation=citation, chunk=chunk, clause_number=number, heading=heading, offset=offset),
            )

    return table


def citable_citation_texts(chunks) -> list[str]:
    return [entry.citation for entry in citable_clauses(chunks).values()]


def citation_grounding(answer: str, chunks, skip_records: bool = False) -> float:
    """The share of the answer's substantive sentences that cite a retrieved extract (0..1).

    A sentence is grounded when it carries a [Document §clause] marker that resolves to one of the
    extracts (or a heading inside one). Sentences with no citation, or citing something that was
    never retrieved, count against the score. An answer with no substantive sentence scores 1.0
    when it cites something and 0.0 when it does not.

    ``skip_records`` leaves out the sentences that report database rows (an answer written over
    both sources): they are sourced by the as-of date, not by a clause. Counted, the 21 vendor lines
    of a hybrid answer put its one cited policy sentence at 0.04.
    """
    citable = citable_clauses(chunks)
    sentences = _substantive_sentences(answer or "")

    if skip_records:
        sentences = [sentence for sentence in sentences if not is_record_sentence(sentence)]

    if not sentences:
        return 1.0 if any(canonical_citation(c) in citable for c in CITATION_PATTERN.findall(answer or "")) else 0.0

    grounded = 0

    for sentence in sentences:
        if any(canonical_citation(citation) in citable for citation in CITATION_PATTERN.findall(sentence)):
            grounded += 1

    return round(grounded / len(sentences), 4)


# ---------------------------------------------------------------------------------------------
# a citation on every claim of a RAG answer (src/nodes/rag_path.py, src/nodes/validation.py)
# ---------------------------------------------------------------------------------------------

_CITATION_MARKER = r"\[[^\]\n]+?§[^\]\n]+?\]"

# "X. [Doc §5] Y." - a marker written after the full stop belongs to the sentence before it
_MARKERS_AFTER_STOP = re.compile(
    rf"(?<=\S)([.!?])[ \t]*({_CITATION_MARKER}(?:[ \t]*{_CITATION_MARKER})*)([ \t]*)"
)

# the answer is read one claim at a time: a sentence, or a line (a bullet or numbered item often has
# no closing full stop). An abbreviation's full stop does not end a sentence, and "Art." / "No." /
# "Sec." only hold it open when a number follows ("Art. 5") - "the state of the art." ends one.
_ALWAYS_OPEN_AFTER = ("e.g.", "i.e.", "etc.", "vs.", "cf.", "u.s.", "u.k.", "e.u.")

_OPEN_BEFORE_NUMBER = ("art.", "no.", "sec.")

CLAIM_BOUNDARY = re.compile(
    "((?<=[.!?])"
    + "".join(rf"(?<!\b{re.escape(abbreviation)})" for abbreviation in _ALWAYS_OPEN_AFTER)
    + "".join(rf"(?!(?<=\b{re.escape(abbreviation)})[ \t]+\d)" for abbreviation in _OPEN_BEFORE_NUMBER)
    + r"[ \t]+|[ \t]*\n+[ \t]*)",
    re.I,
)

_ENDS_WITH_ABBREVIATION = re.compile(r"\b(?:e\.g|i\.e|etc|vs|cf|u\.s|u\.k|e\.u)\.$", re.I)

_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+")

# a markdown heading, a line that is only bold text, and a table row are layout, not claims
_HEADING_LINE = re.compile(r"^(?:#{1,6}\s|(?:\*\*|__)[^*_]+(?:\*\*|__):?\s*$)")

_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")

COURTESY_PREFIXES = ("for further", "for more", "if you have", "feel free", "reach out", "let me know", "i hope", "hope this")

MIN_CLAIM_WORDS = 4

# a line that is only a reference ("- Anti Bribery Policy §4") is a source list, not a claim
REFERENCE_LINE_WORDS = 12

CONTENT_WORD = re.compile(r"[a-z0-9][a-z0-9-]{3,}")

# words every policy extract and every answer share, or that this corpus uses everywhere; they say
# nothing about which clause supports a sentence
GENERIC_WORDS = frozenset(
    {
        "about", "according", "after", "also", "answer", "apply", "applies", "before", "been", "being",
        "between", "both", "clause", "clauses", "company", "compliance", "contractor", "contractors",
        "document", "documents", "does", "during", "each", "either", "employee", "employees", "extract",
        "extracts", "following", "from", "gift", "gifts", "have", "including", "into", "made", "make",
        "more", "most", "must", "only", "other", "over", "policies", "policy", "question", "said", "says",
        "section", "sections", "shall", "should", "some", "staff", "such", "supplier", "suppliers",
        "than", "that", "their", "them", "then", "there", "these", "they", "this", "those", "through",
        "under", "upon", "vendor", "vendors", "very", "were", "what", "when", "where", "which", "while",
        "will", "with", "within", "without", "would", "your",
    }
)

_APOSTROPHE = "['" + chr(0x2019) + "]"

# the evidence as the subject of a sentence: "the extracts", "the provided policy extracts", "the policy"
_EVIDENCE = (
    r"(?:the|these|those|this|its)\s+(?:(?:provided|retrieved|available|supplied|cited|above|given|relevant"
    r"|searched)\s+)?(?:policy\s+|policies\s+)?(?:polic(?:y|ies)|extracts?|excerpts?|documents?|clauses?"
    r"|sources?|evidence|context|text|records)"
)

# "not explicitly detailed", "does not specifically address"
_ADVERB = r"(?:(?:explicitly|clearly|specifically|directly|expressly|fully|further|separately)\s+)?"

# a sentence that only says what the *evidence* does not settle is a disclosure, not a claim: it
# needs no citation and is never given one
GAP_STATEMENT = re.compile(
    rf"\bi\s+(?:do\s+not|don{_APOSTROPHE}t|cannot|can{_APOSTROPHE}t|could\s+not|couldn{_APOSTROPHE}t)\s+"
    r"(?:know|find|determine|confirm|answer|say)\b"
    rf"|\b{_EVIDENCE}\s+(?:do|does|did)(?:\s+not|n{_APOSTROPHE}t)\s+{_ADVERB}"
    r"(?:specify|state|mention|cover|address|define|say|provide|include|contain|give|set\s+out|answer|explain|detail)\b"
    rf"|\bnot\s+{_ADVERB}(?:specified|stated|mentioned|covered|addressed|defined|provided|included|given|set\s+out"
    r"|detailed|described|outlined)"
    rf"\s+(?:in|by|anywhere\s+in)\s+{_EVIDENCE}\b"
    rf"|\b{_EVIDENCE}\s+(?:is|are)\s+silent\b|\bsilent\s+on\b"
    r"|\bthere\s+(?:is|are)\s+no\s+(?:information|mention|details?|guidance|clause|extract|provision)s?\b"
    r"|\bno\s+(?:information|mention|details?|guidance|clause|extract|provision)s?\s+(?:is\s+|was\s+)?"
    rf"(?:available\s+|given\s+|provided\s+)?(?:in|within)\s+{_EVIDENCE}\b"
    rf"|\bcannot\s+be\s+(?:determined|confirmed|answered|verified)\s+from\s+{_EVIDENCE}\b"
    rf"|\b(?:is|are|remains?)\s+unclear\s+(?:from|in)\s+{_EVIDENCE}\b",
    re.I,
)

OBLIGATION_WORDS = re.compile(
    r"\b(?:must|shall|may\s+not|required|prohibited|forbidden|mandatory|not\s+permitted|not\s+allowed)\b", re.I
)

# "X requires Y, but the policy does not specify Z" is a rule and a gap: each part is read on its own
_CLAUSE_SPLIT = re.compile(r";|,?\s+\b(?:but|although|though|however|whereas|while|yet)\b", re.I)

# "the extracts do not specify whether approval is required": the obligation word sits in the open question
_OPEN_QUESTION = re.compile(r"\b(?:whether|if|how|when|which|what|who|whom)\b.*$", re.I | re.S)

# a clause is placed on a sentence only when it clearly supports it: at least MIN_SHARED content words
# in common, covering at least MIN_COVERAGE of the sentence's content words, more than any clause of
# another extract (the writer's own cited clauses and a sub-clause heading only break ties), and the
# clause does not say the opposite or hold different figures (_consistent)
MIN_SHARED = 2

MIN_COVERAGE = 0.4

_PROHIBITION = re.compile(
    r"\b(?:not\s+(?:be\s+)?(?:permitted|allowed|acceptable|accepted|tolerated)|prohibit(?:s|ed)?|forbid(?:s|den)?"
    r"|ban(?:s|ned)?|must\s+not|may\s+not|shall\s+not|cannot|can\s+not|never|unacceptable|zero\s+tolerance)\b",
    re.I,
)

_PERMISSION = re.compile(
    r"\b(?:permitted|allowed|acceptable|optional|exempt|no\s+approval|without\s+(?:any\s+)?approval|not\s+required"
    r"|need\s+not|may\s+(?:be\s+)?(?:accept|accepted|give|given|offer|offered|pay|paid|keep|kept|share|shared))\b",
    re.I,
)

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")

# a sentence that reports database rows: it is sourced by the as-of date and the record identifiers,
# never by a policy clause
_RECORD_SENTENCE = re.compile(
    r"\bas of\b|\bvendor_\d+|\bthe following (?:vendors?|suppliers?|records?|reviews?|findings?)\b|\b\d+\s+(?:vendors?|suppliers?|records?|rows?|reviews?|findings?)\b"
    r"|\b(?:query|database|row count|rows?)\b|\brecords? (?:show|list|indicate)",
    re.I,
)


def is_record_sentence(sentence: str) -> bool:
    return bool(_RECORD_SENTENCE.search(sentence or ""))


def _markers_inside_sentences(answer: str) -> str:
    """'X. [Doc §5] Y.' -> 'X [Doc §5]. Y.' so the marker belongs to the sentence it was written after."""
    return _MARKERS_AFTER_STOP.sub(lambda match: f" {match.group(2)}{match.group(1)}{match.group(3)}", answer or "")


def claim_units(answer: str) -> list[str]:
    """The answer cut into sentences and lines, the way the citation check reads it."""
    return [part for part in CLAIM_BOUNDARY.split(_markers_inside_sentences(answer))[::2] if part.strip()]


def _claim_text(unit: str) -> str:
    return _LIST_MARKER.sub("", unit or "").strip()


def _is_claim(unit: str) -> bool:
    if _TABLE_ROW.match(unit or ""):
        return False

    text = _claim_text(unit)
    lowered = text.lower()

    if _HEADING_LINE.match(text) or len(text) <= 25 or text.endswith(":"):
        return False

    if lowered.startswith(NON_CLAIM_PREFIXES) or lowered.startswith(COURTESY_PREFIXES):
        return False

    if len(re.findall(r"[A-Za-z]{2,}", text)) < MIN_CLAIM_WORDS:
        return False

    bare_reference = "§" in text and not CITATION_PATTERN.search(text) and len(text.split()) <= REFERENCE_LINE_WORDS

    return not bare_reference


def _names_a_gap(part: str) -> bool:
    return bool(GAP_STATEMENT.search(part)) and not OBLIGATION_WORDS.search(_OPEN_QUESTION.sub("", part))


def is_gap_statement(sentence: str) -> bool:
    """True when the sentence, every part of it, only says what the evidence does not settle."""
    parts = [part.strip(" ,.") for part in _CLAUSE_SPLIT.split(sentence or "")]
    parts = [part for part in parts if part]

    return bool(parts) and all(_names_a_gap(part) for part in parts)


def uncited_sentences(answer: str) -> list[str]:
    """Substantive sentences or lines with no [Document §clause] marker that do not just name a gap."""
    return [
        unit.strip()
        for unit in claim_units(answer)
        if _is_claim(unit) and not CITATION_PATTERN.search(unit) and not is_gap_statement(unit)
    ]


def only_names_gaps(answer: str) -> bool:
    """True when nothing in the answer is a claim: every sentence only says what the evidence does not settle."""
    units = claim_units(answer)
    claims = [unit for unit in units if _is_claim(unit) or CITATION_PATTERN.search(unit)]

    if not claims:
        return is_gap_statement(answer or "")

    return all(is_gap_statement(unit) and not CITATION_PATTERN.search(unit) for unit in claims)


def cites_retrieved_clause(answer: str, cited_clauses: list[str], chunks) -> bool:
    table = citable_clauses(chunks)

    return any(
        canonical_citation(raw) in table for raw in [*(cited_clauses or []), *CITATION_PATTERN.findall(answer or "")]
    )


def _content_words(text: str) -> set[str]:
    return {word for word in CONTENT_WORD.findall((text or "").casefold()) if word not in GENERIC_WORDS}


def _is_sub_clause(entry: CitableClause) -> bool:
    return entry.clause_number != getattr(entry.chunk, "clause_number", None)


def _is_whole_document_parent(entry: CitableClause) -> bool:
    return not _is_sub_clause(entry) and is_whole_document_section(entry.chunk) and bool(sub_clause_headings(entry.chunk))


def _same_clause_family(first: CitableClause, second: CitableClause) -> bool:
    """Two entries for the same clause: one extract split in two, or a clause and a heading inside it."""
    if getattr(first.chunk, "document_title", None) != getattr(second.chunk, "document_title", None):
        return False

    a, b = first.clause_number or "", second.clause_number or ""

    return (
        a == b
        or a.startswith(f"{b}.")
        or b.startswith(f"{a}.")
        or is_whole_document_section(first.chunk)
        or is_whole_document_section(second.chunk)
    )


def _clause_text(entry: CitableClause) -> str:
    content = getattr(entry.chunk, "content", "") or ""

    if not _is_sub_clause(entry):
        return f"{entry.heading} {content}"

    # a sub-clause is scored on its own text, from its heading to the next heading in the extract
    later = [offset for _number, _heading, offset in sub_clause_headings(entry.chunk) if offset > entry.offset]
    end = min(later) if later else len(content)

    return f"{entry.heading} {content[entry.offset:end]}"


def _clause_texts(table: dict[str, CitableClause], chunks) -> dict[str, str]:
    texts = {key: _clause_text(entry) for key, entry in table.items()}

    # a clause split across several extracts is read on all of them, not only the first
    for chunk in chunks or []:
        key = canonical_citation(chunk.citation)

        if key in texts and chunk is not table[key].chunk:
            texts[key] = f"{texts[key]} {chunk.section} {chunk.content}"

    return texts


def _polarity(text: str) -> set[str]:
    marks = set()

    if _PROHIBITION.search(text):
        marks.add("prohibit")

    if _PERMISSION.search(_PROHIBITION.sub(" ", text)):
        marks.add("permit")

    return marks


def _consistent(claim: str, clause_text: str) -> bool:
    """The clause does not say the opposite of the claim, and it holds every figure the claim states.

    Word overlap alone placed "[AB §3.1]" on "Cash gifts are permitted when a manager approves them"
    although §3.1 lists cash as prohibited. A claim that permits something is never given a clause
    that prohibits anything, a claim that prohibits is only given a clause that prohibits, and a
    figure the clause does not contain means the clause is not its source. Such a claim stays bare
    and goes back to the writer.
    """
    claim_marks, clause_marks = _polarity(claim), _polarity(clause_text)

    if "permit" in claim_marks and "prohibit" not in claim_marks and "prohibit" in clause_marks:
        return False

    if "prohibit" in claim_marks and "permit" not in claim_marks and "prohibit" not in clause_marks:
        return False

    figures = set(_NUMBER.findall(clause_text))

    return all(figure in figures for figure in _NUMBER.findall(claim))


def _placed_clause(
    claim: str,
    table: dict[str, CitableClause],
    texts: dict[str, str],
    vocabulary: dict[str, set[str]],
    primary: set[str],
    with_parents: bool,
) -> CitableClause | None:
    words = _content_words(claim)
    scored = []

    for key, entry in table.items():
        if not with_parents and _is_whole_document_parent(entry):
            continue

        if "§" not in entry.citation:
            # an extract with no clause number is cited by its title alone, which is not a
            # [Document §clause] marker - placing it would leave the sentence as uncited as before
            continue

        shared = len(words & vocabulary[key])

        if shared:
            scored.append((shared, key in primary, _is_sub_clause(entry), key, entry))

    if not scored:
        return None

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    shared, primary_hit, _specific, best_key, best = scored[0]

    if shared < MIN_SHARED or shared < MIN_COVERAGE * len(words):
        return None

    for other_shared, other_primary, _other_specific, _other_key, other in scored[1:]:
        if (other_shared, other_primary) != (shared, primary_hit):
            break

        if not _same_clause_family(best, other):
            # two different clauses support the sentence equally: which one it rests on is the
            # writer's call, so the sentence is left for the validator to send back
            return None

    return best if _consistent(claim, texts[best_key]) else None


def _supporting_clause(
    claim: str,
    table: dict[str, CitableClause],
    texts: dict[str, str],
    vocabulary: dict[str, set[str]],
    primary: set[str],
) -> CitableClause | None:
    # a whole-document "General §1" extract lends the sub-clause that supports the sentence; its own
    # number is used only when no sub-clause does
    return _placed_clause(claim, table, texts, vocabulary, primary, with_parents=False) or _placed_clause(
        claim, table, texts, vocabulary, primary, with_parents=True
    )


def _attach(unit: str, citation: str) -> str:
    body = unit.rstrip()
    tail = unit[len(body):]
    tag = f" [{citation}]"

    if body and body[-1] in ".!?:" and not _ENDS_WITH_ABBREVIATION.search(body):
        return f"{body[:-1]}{tag}{body[-1]}{tail}"

    return f"{body}{tag}{tail}"


def cite_uncited_sentences(
    answer: str, cited_clauses: list[str], chunks, skip_records: bool = False
) -> tuple[str, list[str]]:
    """A policy answer with a [Document §clause] marker on every claim a clause clearly supports.

    The writer (gpt-4o-mini on this deployment) cites one clause and leaves the other sentences
    bare; the validator then failed the draft with missing_citation, the repair came back the same,
    and a Low-risk policy answer escalated (the anti-bribery trace, twice). Each uncited claim - a
    sentence, or a bullet line - is given the clause that clearly supports it (``_supporting_clause``).
    A claim no clause clearly supports is left bare, and the validator sends it back to the writer:
    nothing here guesses a source for it. A sentence that only names a gap in the evidence is never
    given a citation. Markers the writer put after a full stop are moved inside that sentence.
    """
    table = citable_clauses(chunks)

    if not answer or not table:
        return answer, []

    answer = _markers_inside_sentences(answer)
    primary = {
        key
        for key in (canonical_citation(raw) for raw in [*(cited_clauses or []), *CITATION_PATTERN.findall(answer)])
        if key in table
    }
    texts = _clause_texts(table, chunks)
    vocabulary = {key: _content_words(text) for key, text in texts.items()}

    parts = CLAIM_BOUNDARY.split(answer)
    added: list[str] = []

    for index in range(0, len(parts), 2):
        unit = parts[index]

        if not _is_claim(unit) or CITATION_PATTERN.search(unit) or is_gap_statement(unit):
            continue

        if skip_records and is_record_sentence(unit):
            # a hybrid answer's sentence about the rows is sourced by the as-of date, not a clause
            continue

        chosen = _supporting_clause(_claim_text(unit), table, texts, vocabulary, primary)

        if chosen is None:
            continue

        parts[index] = _attach(unit, chosen.citation)
        added.append(chosen.citation)

    return "".join(parts), added


def with_inline_citations(answer: str, cited_clauses: list[str], chunks) -> tuple[str, list[str]]:
    """The answer with every clause it forgot to mark inline appended, and which ones were added.

    RAG_ANSWER, HYBRID_ANSWER and the panel prompts all ask the model for two things that should
    agree: a [Document Title §clause] marker on every substantive sentence, and the same identifier
    copied into cited_clauses. On this deployment (gpt-4o-mini serving both the small and strong
    tier) the model reliably fills cited_clauses but regularly leaves the marker out of the prose -
    the log showed cited=1 on both the first draft and the widened repair, yet the compliance
    validator failed both with missing_citation because extract_citations(answer) found nothing to
    check. The claim really is grounded; the marker just never made it into the text the validator
    and the output guardrail actually read. Since the clause identifier is already known and already
    resolves to a retrieved extract, it is stitched into the answer here instead of spending another
    repair pass asking the model to remember the same instruction twice.
    """
    table = citable_clauses(chunks)
    present = {canonical_citation(c) for c in CITATION_PATTERN.findall(answer or "")}
    seen: set[str] = set()
    missing: list[str] = []

    for raw in cited_clauses or []:
        key = canonical_citation(raw)

        if key in present or key in seen or key not in table:
            continue

        seen.add(key)
        missing.append(table[key].citation)

    if not missing:
        return answer, []

    tag = " " + " ".join(f"[{citation}]" for citation in missing)
    stripped = (answer or "").rstrip()

    # folded into the closing punctuation of the last sentence so the marker lands on the claim it
    # supports rather than after it - citation_grounding() scores per sentence, and a citation that
    # arrives as its own trailing fragment would not ground the sentence that actually made the claim
    if stripped and stripped[-1] in ".!?":
        fixed = stripped[:-1] + tag + stripped[-1]
    else:
        fixed = stripped + tag

    return fixed, missing
