"""Smoke-test the external MCP server the agent is configured to use.

Run it before a demo:

    uv run python scripts/mcp_smoke.py "GDPR article 33 breach notification 72 hours"

Works for both MCP_SERVER_URL forms: a command (``duckduckgo-mcp-server``, spawned over
stdio by the client) or an ``http://.../mcp`` URL of a server you started separately.

Three checks, in order, so a failure names the layer that broke:

1. plain HTTP reachability with ``httpx`` (is anything listening at MCP_SERVER_URL?)
2. MCP handshake + tool discovery through the same ``McpToolProvider`` the agent uses
   (does the server speak MCP, and do the allowed tools exist?)
3. one real call of the first allowed tool, through the output clamp.
"""

import asyncio
import sys

import httpx

from configs.settings import settings
from src.tools.mcp_tools import get_mcp_provider


def check_http(url: str) -> None:
    if not url.startswith(("http://", "https://")):
        print(f"[1/3] '{url}' is a command, not a URL - the client will spawn it over stdio")
        return

    with httpx.Client(timeout=5.0) as client:
        try:
            response = client.get(url)
        except httpx.ConnectError as exc:
            sys.exit(f"[1/3] nothing is listening at {url}: {exc}")

    print(f"[1/3] {url} is reachable (HTTP {response.status_code})")


async def check_mcp(query: str) -> None:
    provider = get_mcp_provider()

    if not provider.configured:
        sys.exit("[2/3] MCP_SERVER_URL is empty; set it in .env first")

    tools = await provider.load_tools()

    if not tools:
        sys.exit("[2/3] the server answered but exposed none of the allowed tools: " f"{provider.allowed}")

    names = [tool.metadata.name for tool in tools]
    print(f"[2/3] MCP handshake ok, tools loaded: {names}")

    first = tools[0]
    kwargs = {"query": query} if first.metadata.name == "search" else {"url": query}

    result = await first.acall(**kwargs)
    text = str(getattr(result, "content", result))

    print(f"[3/3] {first.metadata.name}({kwargs}) returned {len(text)} chars:\n")
    print(text[:1200])


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "GDPR article 33 personal data breach notification deadline"

    check_http(settings.mcp_server_url)
    asyncio.run(check_mcp(question))
