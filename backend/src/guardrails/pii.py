import logging
import re

logger = logging.getLogger(__name__)

FALLBACK_PATTERNS = {
    "EMAIL_ADDRESS": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "PHONE_NUMBER": re.compile(r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "AADHAAR": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "PAN": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
}

SUPPORTED_PRESIDIO_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "PERSON",
    "LOCATION",
    "IP_ADDRESS",
]

_analyzer = None
_anonymizer = None


def _load_presidio():
    global _analyzer, _anonymizer

    if _analyzer is not None:
        return _analyzer, _anonymizer

    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        _analyzer = AnalyzerEngine()
        _anonymizer = AnonymizerEngine()
    except Exception as exc:
        logger.warning("presidio unavailable, falling back to regex redaction: %s", exc)
        _analyzer = False
        _anonymizer = False

    return _analyzer, _anonymizer


def _regex_redact(text: str) -> tuple[str, list[str]]:
    found: list[str] = []
    redacted = text

    for label, pattern in FALLBACK_PATTERNS.items():
        if pattern.search(redacted):
            found.append(label)
            redacted = pattern.sub(f"<{label}>", redacted)

    return redacted, found


def redact(text: str) -> tuple[str, list[str]]:
    analyzer, anonymizer = _load_presidio()

    if not analyzer:
        return _regex_redact(text)

    try:
        results = analyzer.analyze(text=text, entities=SUPPORTED_PRESIDIO_ENTITIES, language="en")

        if not results:
            return _regex_redact(text)

        anonymised = anonymizer.anonymize(text=text, analyzer_results=results)
        entity_types = sorted({item.entity_type for item in results})

        merged, extra = _regex_redact(anonymised.text)

        return merged, sorted(set(entity_types) | set(extra))
    except Exception as exc:
        logger.warning("presidio analysis failed, using regex fallback: %s", exc)
        return _regex_redact(text)


def contains_pii(text: str) -> bool:
    _, found = redact(text)

    return bool(found)
