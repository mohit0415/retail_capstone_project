import re
import unicodedata

CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

BINARY_SIGNATURES = ("%PDF-", "PK\x03\x04", "\xff\xd8\xff", "\x89PNG")


def clean_extracted_text(text: str) -> str:
    if not text:
        return ""

    normalised = unicodedata.normalize("NFC", text)

    return CONTROL_CHARACTERS.sub("", normalised.replace("\x00", ""))


def assert_not_raw_bytes(text: str, file_name: str) -> None:
    head = (text or "").lstrip()[:8]

    if head.startswith(BINARY_SIGNATURES):
        raise ValueError(
            f"the reader returned the raw bytes of {file_name} instead of its text, which means "
            f"the LlamaIndex file reader is not installed. Run 'uv add llama-index-readers-file' "
            f"in backend/ and ingest again."
        )
