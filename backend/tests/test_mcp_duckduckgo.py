"""Live integration test for the external DuckDuckGo MCP server.

The unit tests in ``test_tools.py`` replace the MCP client, so they never prove
that a real server is wired in. This module does, and it skips itself unless
``MCP_SERVER_URL`` is set *and* something is listening there, so CI and a
laptop without the server still pass.

With ``MCP_SERVER_URL=duckduckgo-mcp-server`` (stdio) it only needs the package installed
(``uv add duckduckgo-mcp-server``); with an ``http://`` URL, start that server first.
"""

import asyncio
import shutil

import httpx
import pytest

from configs.settings import settings
from src.tools.mcp_tools import McpToolProvider


def _server_is_up(url: str) -> bool:
    if not url:
        return False

    if not url.startswith(("http://", "https://")):
        return shutil.which(url) is not None

    try:
        httpx.get(url, timeout=3.0)
    except httpx.HTTPError:
        return False

    return True


pytestmark = pytest.mark.skipif(
    not _server_is_up(settings.mcp_server_url),
    reason="no MCP server listening at MCP_SERVER_URL",
)


@pytest.fixture(scope="module")
def mcp_tools():
    return asyncio.run(McpToolProvider().load_tools())


def test_the_allowed_tools_are_discovered(mcp_tools):
    names = {tool.metadata.name for tool in mcp_tools}
    allowed = {name.strip() for name in settings.mcp_allowed_tools.split(",") if name.strip()}

    assert names == allowed, f"server exposed {names}, .env allows {allowed}"


def test_tools_outside_the_allowlist_are_not_loaded(mcp_tools):
    names = {tool.metadata.name for tool in mcp_tools}

    assert not (names - {"search", "fetch_content"})


def test_a_search_returns_clamped_text(mcp_tools):
    search = next(tool for tool in mcp_tools if tool.metadata.name == "search")

    result = asyncio.run(search.acall(query="GDPR article 33 breach notification 72 hours"))
    text = str(getattr(result, "content", result))

    assert "72" in text or "seventy-two" in text.lower()
    assert len(text) <= settings.mcp_max_output_chars + 120
