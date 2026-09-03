import logging
from typing import List

from llama_index.core.tools import BaseTool, FunctionTool

from configs.settings import settings

logger = logging.getLogger(__name__)


def _clamp(tool: BaseTool, max_chars: int) -> FunctionTool:
    async def clamped(**kwargs):
        result = await tool.acall(**kwargs)

        raw = getattr(result, "raw_output", None)
        contents = getattr(raw, "content", None) if raw is not None else None

        if contents:
            text = "\n".join(getattr(item, "text", "") or "" for item in contents)
        else:
            text = str(getattr(result, "content", result))

        if len(text) > max_chars:
            text = (
                text[:max_chars]
                + f"\n[output truncated at {max_chars} characters - narrow the query rather than refetching]"
            )

        return text

    return FunctionTool.from_defaults(async_fn=clamped, tool_metadata=tool.metadata)


class McpToolProvider:
    def __init__(self):
        self.url = settings.mcp_server_url
        self.allowed = [name.strip() for name in settings.mcp_allowed_tools.split(",") if name.strip()]
        self.timeout = settings.mcp_timeout_seconds

    @property
    def configured(self) -> bool:
        return bool(self.url)

    async def load_tools(self) -> List[BaseTool]:
        if not self.configured:
            logger.info("MCP_SERVER_URL is not set; the agent runs without external MCP tools")
            return []

        try:
            from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

            client = BasicMCPClient(command_or_url=self.url, timeout=self.timeout)

            spec = McpToolSpec(client=client, allowed_tools=self.allowed or None)

            tools = await spec.to_tool_list_async()

            logger.info(
                "loaded %d MCP tool(s) from %s: %s",
                len(tools),
                self.url,
                [tool.metadata.name for tool in tools],
            )

            return [_clamp(tool, settings.mcp_max_output_chars) for tool in tools]
        except Exception as exc:
            logger.error("MCP tools could not be loaded from %s: %s", self.url, exc)

            if settings.mcp_required:
                raise ConnectionError(f"MCP tools are required but could not be loaded: {exc}") from exc

            return []


_provider: McpToolProvider | None = None


def get_mcp_provider() -> McpToolProvider:
    global _provider

    if _provider is None:
        _provider = McpToolProvider()

    return _provider
