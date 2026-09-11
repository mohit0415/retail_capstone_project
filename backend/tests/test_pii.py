"""PII redaction in ``src/guardrails/pii.py``.

Presidio is optional at runtime; these tests pin the regex fallback that every
deployment relies on when it is absent, and the seam that chooses between the
two so a broken Presidio never lets PII through.
"""

import pytest

from src.guardrails import pii


@pytest.fixture
def regex_only(monkeypatch):
    """Force the regex fallback regardless of what is installed."""
    monkeypatch.setattr(pii, "_analyzer", False)
    monkeypatch.setattr(pii, "_anonymizer", False)


@pytest.mark.parametrize(
    ("text", "label", "leaked"),
    [
        ("contact john.doe@example.com about the breach", "EMAIL_ADDRESS", "john.doe@example.com"),
        ("call the DPO on 9876543210 today", "PHONE_NUMBER", "9876543210"),
        ("call the DPO on +91-9876543210 today", "PHONE_NUMBER", "9876543210"),
        ("card 4111 1111 1111 1111 was charged", "CREDIT_CARD", "4111 1111 1111 1111"),
        ("aadhaar 2345 6789 0123 on file", "AADHAAR", "2345 6789 0123"),
        ("PAN ABCDE1234F belongs to the vendor", "PAN", "ABCDE1234F"),
        ("refund to GB82WEST12345698765432 please", "IBAN", "GB82WEST12345698765432"),
    ],
)
def test_each_fallback_pattern_redacts_its_entity(regex_only, text, label, leaked):
    redacted, found = pii.redact(text)

    assert label in found
    assert leaked not in redacted
    assert f"<{label}>" in redacted


def test_clean_policy_text_is_left_alone(regex_only):
    text = "Clause 4.2 says invoices are retained for seven years, per ISO 27001 A.8.10."

    assert pii.redact(text) == (text, [])
    assert pii.contains_pii(text) is False


def test_several_entities_in_one_sentence_are_all_reported(regex_only):
    redacted, found = pii.redact("mail a.b@c.io or ring 9123456789")

    assert set(found) == {"EMAIL_ADDRESS", "PHONE_NUMBER"}
    assert "a.b@c.io" not in redacted
    assert "9123456789" not in redacted


def test_a_landline_or_short_number_is_not_mistaken_for_a_mobile(regex_only):
    redacted, found = pii.redact("see clause 12345 and the 2024 review")

    assert found == []
    assert redacted == "see clause 12345 and the 2024 review"


def test_contains_pii_is_true_whenever_anything_was_redacted(regex_only):
    assert pii.contains_pii("write to dpo@retailer.in") is True


def test_the_analyzer_is_loaded_once_and_a_failure_is_remembered(monkeypatch):
    monkeypatch.setattr(pii, "_analyzer", None)
    monkeypatch.setattr(pii, "_anonymizer", None)

    import builtins

    real_import = builtins.__import__
    attempts = []

    def _no_presidio(name, *args, **kwargs):
        if name.startswith("presidio"):
            attempts.append(name)

            raise ImportError("presidio not installed")

        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_presidio)

    assert pii._load_presidio() == (False, False)
    assert pii._load_presidio() == (False, False)
    assert len(attempts) == 1, "a missing Presidio must not be re-imported on every call"


class _Result:
    def __init__(self, entity_type):
        self.entity_type = entity_type


class _Anonymised:
    def __init__(self, text):
        self.text = text


def test_presidio_findings_are_merged_with_the_regex_pass(monkeypatch):
    class Analyzer:
        def analyze(self, text, entities, language):
            assert language == "en"
            assert "PERSON" in entities

            return [_Result("PERSON")]

    class Anonymizer:
        def anonymize(self, text, analyzer_results):
            return _Anonymised(text.replace("Priya Sharma", "<PERSON>"))

    monkeypatch.setattr(pii, "_analyzer", Analyzer())
    monkeypatch.setattr(pii, "_anonymizer", Anonymizer())

    redacted, found = pii.redact("Priya Sharma has PAN ABCDE1234F")

    assert redacted == "<PERSON> has PAN <PAN>"
    assert found == ["PAN", "PERSON"]


def test_when_presidio_finds_nothing_the_regex_pass_still_runs(monkeypatch):
    class Analyzer:
        def analyze(self, **kwargs):
            return []

    monkeypatch.setattr(pii, "_analyzer", Analyzer())
    monkeypatch.setattr(pii, "_anonymizer", object())

    redacted, found = pii.redact("mail dpo@retailer.in")

    assert found == ["EMAIL_ADDRESS"]
    assert "dpo@retailer.in" not in redacted


def test_a_presidio_crash_falls_back_instead_of_leaking(monkeypatch):
    class Analyzer:
        def analyze(self, **kwargs):
            raise RuntimeError("spacy model missing")

    monkeypatch.setattr(pii, "_analyzer", Analyzer())
    monkeypatch.setattr(pii, "_anonymizer", object())

    redacted, found = pii.redact("mail dpo@retailer.in")

    assert found == ["EMAIL_ADDRESS"]
    assert "<EMAIL_ADDRESS>" in redacted
