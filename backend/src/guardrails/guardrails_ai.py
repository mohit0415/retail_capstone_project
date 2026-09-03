import logging
import threading
from typing import List, Tuple

from configs.settings import settings

logger = logging.getLogger(__name__)

DEFAULT_PII_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_SSN",
    "US_BANK_NUMBER",
    "PERSON",
    "LOCATION",
]

INSTALL_HINT = (
    "Guardrails Hub validators are not installed. Run:\n"
    "  guardrails configure\n"
    "  guardrails hub install hub://guardrails/detect_pii\n"
    "The system falls back to the Presidio and regex scrub in src/guardrails/pii.py until then."
)


class GuardrailsAiScrubber:
    def __init__(self):
        self.available = False
        self.guard = None

        if not settings.enable_guardrails_ai:
            return

        try:
            from guardrails import Guard
            from guardrails.hub import DetectPII

            configured = [item.strip() for item in settings.pii_entities.split(",") if item.strip()]
            entities = configured or DEFAULT_PII_ENTITIES

            self.guard = Guard().use(DetectPII(pii_entities=entities, on_fail="fix", use_local=True))
            self.available = True

            logger.info("Guardrails AI DetectPII active for entities: %s", entities)
        except ImportError:
            logger.warning(INSTALL_HINT)
        except Exception as exc:
            logger.warning("Guardrails AI could not be initialised (%s); using the local fallback", exc)

    def scrub(self, text: str) -> Tuple[bool, str, List[str]]:
        if not self.available or not text:
            return True, text, []

        try:
            result = self.guard.validate(text)
            scrubbed = result.validated_output or text

            if scrubbed != text:
                return True, scrubbed, ["guardrails_detect_pii"]

            return True, scrubbed, []
        except Exception as exc:
            logger.error("PII validation failed; withholding the output: %s", exc)

            return False, (
                "The response was withheld because the PII safety check could not be completed."
            ), [f"guardrails_failure:{exc}"]


_scrubber: GuardrailsAiScrubber | None = None
_lock = threading.Lock()


def get_scrubber() -> GuardrailsAiScrubber:
    global _scrubber

    if _scrubber is None:
        with _lock:
            if _scrubber is None:
                _scrubber = GuardrailsAiScrubber()

    return _scrubber
