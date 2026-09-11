"""
Check that the Langfuse dashboard prompts and ``src/prompts/library.py`` agree.

    uv run python scripts/check_prompts_sync.py            # report only
    uv run python scripts/check_prompts_sync.py --push     # create missing / changed ones in Langfuse
    uv run python scripts/check_prompts_sync.py --pull     # overwrite library.py with the dashboard text

For every entry in ``PROMPT_REGISTRY`` the script fetches the ``production``
version from Langfuse and compares it with the local template. It reports:

    OK        dashboard text == local text
    MISSING   no prompt with that name in Langfuse (the app is using the local fallback)
    DIFF      both exist but the text differs (shows a unified diff)
    VARS      placeholders differ - the code passes variables the dashboard prompt
              does not use, or the dashboard uses ones the code never supplies

``--push`` makes Langfuse match the code; ``--pull`` makes the code match
Langfuse. Run without either flag first and read the report.
"""

import argparse
import difflib
import logging
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.settings import settings  # noqa: E402
from src.prompts import library  # noqa: E402
from src.prompts.langfuse_prompts import PROMPT_REGISTRY  # noqa: E402

logging.basicConfig(level=logging.WARNING)

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
LIBRARY_PATH = ROOT / "src" / "prompts" / "library.py"


def _client():
    public_key = settings.langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = settings.langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    host = settings.langfuse_host or os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com"

    if not (public_key and secret_key):
        sys.exit("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set in .env")

    from langfuse import Langfuse

    return Langfuse(public_key=public_key, secret_key=secret_key, host=host)


def _remote_text(client, name: str) -> str | None:
    try:
        prompt = client.get_prompt(name, label="production", cache_ttl_seconds=0)
    except Exception:
        return None

    text = getattr(prompt, "prompt", None)

    return text if isinstance(text, str) else None


def _vars(text: str) -> set[str]:
    return set(PLACEHOLDER.findall(text))


def _replace_constant(source: str, attr: str, new_text: str) -> str:
    pattern = re.compile(rf'^{attr} = """(.*?)"""', re.S | re.M)

    if not pattern.search(source):
        raise ValueError(f"{attr} not found in library.py")

    return pattern.sub(lambda _m: f'{attr} = """{new_text}"""', source, count=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--push", action="store_true", help="write local templates to Langfuse")
    group.add_argument("--pull", action="store_true", help="write Langfuse text into library.py")
    args = parser.parse_args()

    client = _client()
    library_source = LIBRARY_PATH.read_text()
    problems = 0

    for name, attr in PROMPT_REGISTRY.items():
        local = getattr(library, attr)
        remote = _remote_text(client, name)

        if remote is None:
            status = "MISSING"
        elif remote == local:
            status = "OK"
        elif _vars(remote) != _vars(local):
            status = "VARS"
        else:
            status = "DIFF"

        print(f"{status:8} {name}")

        if status == "OK":
            continue

        problems += 1

        if status == "VARS":
            print(f"         local vars : {sorted(_vars(local))}")
            print(f"         remote vars: {sorted(_vars(remote))}")

        if status in ("DIFF", "VARS"):
            diff = difflib.unified_diff(
                local.splitlines(), remote.splitlines(), "library.py", "langfuse", lineterm="", n=1
            )
            for line in diff:
                print("         " + line)

        if args.push:
            client.create_prompt(name=name, type="text", prompt=local, labels=["production"])
            print(f"         -> pushed local text to Langfuse as new '{name}' version")
        elif args.pull and remote is not None:
            library_source = _replace_constant(library_source, attr, remote)
            print(f"         -> pulled Langfuse text into library.{attr}")

    if args.pull:
        LIBRARY_PATH.write_text(library_source)

    if args.push:
        client.flush()

    print()
    print(f"{len(PROMPT_REGISTRY) - problems}/{len(PROMPT_REGISTRY)} prompts in sync")

    return 0 if problems == 0 or args.push or args.pull else 1


if __name__ == "__main__":
    raise SystemExit(main())
