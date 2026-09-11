"""Compatibility shim - the real configuration lives in ``src.observability.logging_config``.

Earlier revisions of this file called ``logging.basicConfig(force=True)`` at
import time with its own format and file location, which silently replaced
the handlers ``configure_logging`` had installed (dropping the rotating
``logs/rpids.log`` and the request-id stamp) for any module that happened to
import it. Nothing in the tree imports it any more, but scripts written
against the old name still work: ``setup_logging`` and ``logger`` are kept
and delegate to the central configuration.

Prefer::

    import logging
    logger = logging.getLogger(__name__)

in application code, and ``configure_logging()`` once in the entry point.
"""

import logging

from src.observability.logging_config import configure_logging


def setup_logging(level: int | str = logging.INFO) -> logging.Logger:
    """Install the app-wide logging configuration (idempotent)."""
    name = logging.getLevelName(level) if isinstance(level, int) else str(level)

    return configure_logging(name)


logger = logging.getLogger("rpids")

__all__ = ["logger", "setup_logging"]
