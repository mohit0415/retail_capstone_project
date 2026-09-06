import asyncio
import logging

from llama_index.core.tools import BaseTool

from src.auth.rbac import allowed_doc_types, allowed_risk_categories, allowed_tables
from src.tools.mcp_tools import get_mcp_provider
from src.tools.nl2sql_tool import build_sql_tool
from src.tools.policy_tool import build_policy_tool

logger = logging.getLogger(__name__)


def _load_mcp_tools() -> list[BaseTool]:
    provider = get_mcp_provider()

    if not provider.configured:
        return []

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    try:
        if loop is not None and loop.is_running():
            logger.info("event loop already running; MCP tools are loaded on a worker thread")

            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, provider.load_tools()).result(
                    timeout=provider.timeout + 5
                )

        return asyncio.run(provider.load_tools())
    except Exception as exc:
        logger.error("MCP tool loading failed: %s", exc)
        return []


def build_tools_for(
    access_scopes: list[str],
    departments: list[str] | None = None,
    doc_scope: list[str] | None = None,
    include_mcp: bool = True,
) -> list[BaseTool]:
    documents = allowed_doc_types(access_scopes)
    tables = allowed_tables(access_scopes)

    tools: list[BaseTool] = []

    if documents:
        try:
            tools.append(build_policy_tool(documents, doc_scope))
        except Exception as exc:
            logger.error("policy tool could not be built: %s", exc)

    if tables:
        tools.append(
            build_sql_tool(
                allowed_tables=tables,
                departments=departments or [],
                risk_categories=allowed_risk_categories(access_scopes),
            )
        )

    if include_mcp:
        tools.extend(_load_mcp_tools())

    return tools


def tool_names(tools: list[BaseTool]) -> list[str]:
    return [tool.metadata.name for tool in tools]
