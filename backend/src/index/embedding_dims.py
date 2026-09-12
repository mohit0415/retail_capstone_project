"""Embedding deployment -> vector width.

The embedding model is picked on the login page, so the dimension cannot stay a
static .env value: text-embedding-3-large writes 3072-wide vectors and
text-embedding-3-small writes 1536-wide ones, and asking Azure for the wrong
width is rejected at embed time. ``dimensions_for_deployment`` derives the width
from the deployment name at the moment the credentials are applied.

Azure deployment names are chosen by whoever created the resource, so the match
is on the substring rather than the exact model id ("kb-embed-3-large" resolves
the same as "text-embedding-3-large").
"""

import logging

logger = logging.getLogger(__name__)

# longest marker first: "3-large" has to win over a bare "large"
DIMENSIONS_BY_MARKER: tuple[tuple[str, int], ...] = (
    ("text-embedding-3-large", 3072),
    ("text-embedding-3-small", 1536),
    ("text-embedding-ada-002", 1536),
    ("3-large", 3072),
    ("3-small", 1536),
    ("ada-002", 1536),
    ("large", 3072),
    ("small", 1536),
)

DEFAULT_DIMENSIONS = 1536


def dimensions_for_deployment(deployment: str, fallback: int | None = None) -> int:
    """Vector width for ``deployment``.

    An unrecognised name keeps ``fallback`` (the width already configured), so a
    custom deployment name never silently re-dimensions the corpus.
    """
    name = (deployment or "").strip().lower()

    for marker, dimensions in DIMENSIONS_BY_MARKER:
        if marker in name:
            return dimensions

    resolved = fallback if fallback else DEFAULT_DIMENSIONS

    logger.warning(
        "embedding deployment %r is not a known model name, keeping dimensions=%d",
        deployment,
        resolved,
    )

    return resolved
