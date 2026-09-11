"""
Push the local prompt templates to Langfuse Prompt Management.

Run once (and again whenever a template in ``src/prompts/library.py`` changes):

    uv run python scripts/register_prompts.py

Each prompt is created as a new version with the ``production`` label, which is
the label ``src.prompts.langfuse_prompts.render_prompt`` fetches at runtime.
Requires LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST in ``.env``.
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.settings import settings
from src.prompts import library
from src.prompts.langfuse_prompts import PROMPT_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("register_prompts")


def main() -> int:
    public_key = settings.langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = settings.langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    host = settings.langfuse_host or os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com"

    if not (public_key and secret_key):
        logger.error("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set")

        return 1

    from langfuse import Langfuse

    client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)

    for prompt_name, attr in PROMPT_REGISTRY.items():
        template = getattr(library, attr)

        client.create_prompt(
            name=prompt_name,
            type="text",
            prompt=template,
            labels=["production"],
            tags=["retail-policy-intelligence"],
        )

        logger.info("registered '%s' from library.%s", prompt_name, attr)

    client.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
