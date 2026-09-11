"""
Prompt management contract tests.

These run without Langfuse. They guarantee that:

* every prompt constant in ``src/prompts/library.py`` is registered and uses
  ``{{...}}`` placeholders only (never Python ``{...}``);
* every ``render_prompt(...)`` call site in ``src/`` passes exactly the
  variables its template declares - a placeholder the code does not supply
  would reach the model as literal ``{{name}}`` text, and a variable the
  template does not use is a silent no-op;
* the local fallback compiles like Langfuse's ``compile()``;
* the Langfuse path is used when a managed prompt is available, and the
  fallback is used when it is not.
"""

import ast
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.prompts import langfuse_prompts, library
from src.prompts.langfuse_prompts import (
    PROMPT_REGISTRY,
    compile_template,
    placeholders,
    render_prompt,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

SINGLE_BRACE = re.compile(r"(?<!\{)\{[A-Za-z_][A-Za-z0-9_]*\}(?!\})")


def _library_constants() -> dict[str, str]:
    return {
        name: value
        for name, value in vars(library).items()
        if name.isupper() and isinstance(value, str)
    }


def _call_sites() -> list[tuple[str, int, str, str, set[str]]]:
    """(file, line, langfuse_name, constant_name, kwargs) for every render_prompt call."""
    sites = []

    for path in SRC.rglob("*.py"):
        if path.name == "langfuse_prompts.py":
            continue

        tree = ast.parse(path.read_text(), filename=str(path))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            func = node.func
            func_name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)

            if func_name != "render_prompt":
                continue

            assert len(node.args) >= 2, f"{path}:{node.lineno} render_prompt needs (name, template, ...)"
            assert isinstance(node.args[0], ast.Constant), f"{path}:{node.lineno} prompt name must be a literal"
            assert isinstance(node.args[1], ast.Name), f"{path}:{node.lineno} template must be a library constant"

            kwargs = {kw.arg for kw in node.keywords if kw.arg is not None}
            has_splat = any(kw.arg is None for kw in node.keywords)

            sites.append((str(path.relative_to(ROOT)), node.lineno, node.args[0].value, node.args[1].id, kwargs, has_splat))

    return sites


def test_every_library_prompt_is_registered():
    constants = set(_library_constants())
    registered = set(PROMPT_REGISTRY.values())

    assert constants == registered, (
        f"unregistered: {sorted(constants - registered)}; "
        f"registered but missing from library: {sorted(registered - constants)}"
    )


def test_registry_names_match_constants():
    for langfuse_name, attr in PROMPT_REGISTRY.items():
        assert langfuse_name == attr


def test_library_uses_double_braces_only():
    for name, text in _library_constants().items():
        assert not SINGLE_BRACE.search(text), f"{name} still contains a Python-style {{placeholder}}"
        assert placeholders(text), f"{name} declares no {{{{variables}}}}"


def test_call_sites_match_templates():
    sites = _call_sites()
    assert sites, "no render_prompt call sites found under src/"

    problems = []

    for file, line, langfuse_name, attr, kwargs, has_splat in sites:
        if langfuse_name != attr:
            problems.append(f"{file}:{line} name {langfuse_name!r} != constant {attr!r}")

        if langfuse_name not in PROMPT_REGISTRY:
            problems.append(f"{file}:{line} {langfuse_name!r} not in PROMPT_REGISTRY")

        if has_splat:
            continue

        expected = placeholders(getattr(library, attr))

        if kwargs != expected:
            problems.append(
                f"{file}:{line} {attr}: missing={sorted(expected - kwargs)} extra={sorted(kwargs - expected)}"
            )

    assert not problems, "\n".join(problems)


def test_compile_template_substitutes_and_leaves_unknown():
    out = compile_template("a {{x}} b {{ y }} c {{missing}}", x="1", y=2)

    assert out == "a 1 b 2 c {{missing}}"


def test_fallback_render_matches_python_format_semantics(no_langfuse):
    # This checks the *local* compile path, so Langfuse must be out of the picture:
    # with LANGFUSE_* keys in .env, render_prompt would otherwise return the managed
    # dashboard text, and any drift there would fail this test for the wrong reason.
    for attr in PROMPT_REGISTRY.values():
        template = getattr(library, attr)
        names = placeholders(template)
        values = {n: f"<{n}>" for n in names}

        rendered = render_prompt(attr, template, **values)
        legacy = re.sub(r"\{\{(\w+)\}\}", r"{\1}", template).format(**values)

        assert rendered == legacy, attr
        assert "{{" not in rendered, attr


@pytest.fixture
def no_langfuse(monkeypatch):
    monkeypatch.setattr(langfuse_prompts, "_client", lambda: None)
    langfuse_prompts._source.clear()


@pytest.fixture
def fake_langfuse(monkeypatch):
    """A client whose get_prompt returns a managed prompt with different text."""
    managed = MagicMock()
    managed.is_fallback = False
    managed.compile.side_effect = lambda **kw: compile_template("MANAGED {{transcript}}", **kw)

    client = MagicMock()
    client.get_prompt.return_value = managed

    monkeypatch.setattr(langfuse_prompts, "_client", lambda: client)
    langfuse_prompts._source.clear()

    return client


def test_render_uses_fallback_when_langfuse_disabled(no_langfuse):
    out = render_prompt("THREAD_SUMMARY", library.THREAD_SUMMARY, transcript="user: hi")

    assert "user: hi" in out
    assert out.startswith("You keep a running summary")
    assert langfuse_prompts.prompt_sources()["THREAD_SUMMARY"] == "fallback"


def test_render_uses_langfuse_when_available(fake_langfuse):
    out = render_prompt("THREAD_SUMMARY", library.THREAD_SUMMARY, transcript="user: hi")

    assert out == "MANAGED user: hi"
    fake_langfuse.get_prompt.assert_called_once()
    assert fake_langfuse.get_prompt.call_args.args[0] == "THREAD_SUMMARY"
    assert fake_langfuse.get_prompt.call_args.kwargs["label"] == "production"
    assert langfuse_prompts.prompt_sources()["THREAD_SUMMARY"] == "langfuse"


def test_render_falls_back_when_sdk_returns_fallback_object(monkeypatch):
    prompt = MagicMock()
    prompt.is_fallback = True
    client = MagicMock()
    client.get_prompt.return_value = prompt
    monkeypatch.setattr(langfuse_prompts, "_client", lambda: client)

    out = render_prompt("THREAD_SUMMARY", library.THREAD_SUMMARY, transcript="X")

    prompt.compile.assert_not_called()
    assert "X" in out and out.startswith("You keep a running summary")


def test_render_falls_back_when_fetch_raises(monkeypatch):
    client = MagicMock()
    client.get_prompt.side_effect = RuntimeError("network down")
    monkeypatch.setattr(langfuse_prompts, "_client", lambda: client)

    out = render_prompt("THREAD_SUMMARY", library.THREAD_SUMMARY, transcript="X")

    assert "X" in out and "{{" not in out


def test_warm_prompt_cache_reports_every_prompt(no_langfuse):
    sources = langfuse_prompts.warm_prompt_cache()

    assert set(sources) == set(PROMPT_REGISTRY)
    assert set(sources.values()) == {"fallback"}
