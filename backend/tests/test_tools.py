"""The agent tool layer in ``src/tools/``.

``registry.build_tools_for`` decides which tools a principal gets from its
access scopes; ``nl2sql_tool`` wraps the vetted-SQL executor for the agent;
``mcp_tools`` guards the optional MCP server. The retriever, the database and
the MCP client are all replaced, so the tests pin the role-to-tool mapping,
the tool's error contract and the output clamp.
"""

import asyncio
import json

import pytest
from llama_index.core.tools import FunctionTool

from configs.settings import settings
from src.auth.rbac import access_scopes_for
from src.schemas.enums import Role
from src.schemas.models import SqlEvidence
from src.sqlpath.executor import SqlPolicyError
from src.tools import mcp_tools, nl2sql_tool, registry


def _tool(name: str) -> FunctionTool:
    return FunctionTool.from_defaults(fn=lambda input: input, name=name, description=name)


@pytest.fixture
def fake_builders(monkeypatch):
    calls = {"policy": [], "sql": [], "mcp": 0}

    def _policy(documents, doc_scope):
        calls["policy"].append((documents, doc_scope))

        return _tool("policy_documents")

    def _sql(allowed_tables, departments, risk_categories):
        calls["sql"].append((allowed_tables, departments, risk_categories))

        return _tool("compliance_records")

    def _mcp():
        calls["mcp"] += 1

        return [_tool("external_lookup")]

    monkeypatch.setattr(registry, "build_policy_tool", _policy)
    monkeypatch.setattr(registry, "build_sql_tool", _sql)
    monkeypatch.setattr(registry, "_load_mcp_tools", _mcp)

    return calls


def test_a_store_associate_gets_only_the_policy_tool(fake_builders):
    tools = registry.build_tools_for(access_scopes_for(Role.STORE_ASSOCIATE), include_mcp=False)

    assert registry.tool_names(tools) == ["policy_documents"]
    assert fake_builders["sql"] == []

    [(documents, doc_scope)] = fake_builders["policy"]
    assert set(documents) == {"privacy_policy", "infosec_policy", "anti_bribery_policy", "retention_policy"}
    assert doc_scope is None


def test_a_store_manager_gets_the_sql_tool_with_its_narrow_grants(fake_builders):
    tools = registry.build_tools_for(
        access_scopes_for(Role.STORE_MANAGER), departments=["Sales"], include_mcp=False
    )

    assert registry.tool_names(tools) == ["policy_documents", "compliance_records"]

    [(tables, departments, risk)] = fake_builders["sql"]
    assert tables == {"vendors", "compliance_reviews"}
    assert departments == ["Sales"]
    assert risk == {"Low", "Medium"}


def test_a_compliance_officer_sees_every_table_and_risk_category(fake_builders):
    registry.build_tools_for(access_scopes_for(Role.COMPLIANCE_OFFICER), include_mcp=False)

    [(tables, _, risk)] = fake_builders["sql"]
    assert tables == {"vendors", "compliance_reviews", "audit_logs", "retention_records"}
    assert risk == {"Low", "Medium", "High", "Critical"}


def test_the_document_scope_is_handed_to_the_policy_tool(fake_builders):
    registry.build_tools_for(access_scopes_for(Role.ADMIN), doc_scope=["gdpr"], include_mcp=False)

    assert fake_builders["policy"][0][1] == ["gdpr"]


def test_mcp_tools_are_appended_only_when_asked_for(fake_builders):
    with_mcp = registry.build_tools_for(access_scopes_for(Role.ADMIN))
    without = registry.build_tools_for(access_scopes_for(Role.ADMIN), include_mcp=False)

    assert registry.tool_names(with_mcp)[-1] == "external_lookup"
    assert "external_lookup" not in registry.tool_names(without)
    assert fake_builders["mcp"] == 1


def test_no_scopes_means_no_tools(fake_builders):
    assert registry.build_tools_for([], include_mcp=False) == []


def test_a_policy_tool_that_cannot_be_built_is_logged_and_dropped(fake_builders, monkeypatch, caplog):
    def _boom(documents, doc_scope):
        raise RuntimeError("vector store unreachable")

    monkeypatch.setattr(registry, "build_policy_tool", _boom)

    with caplog.at_level("ERROR", logger="src.tools.registry"):
        tools = registry.build_tools_for(access_scopes_for(Role.STORE_MANAGER), include_mcp=False)

    assert registry.tool_names(tools) == ["compliance_records"]
    assert "policy tool could not be built" in caplog.text


def test_mcp_loading_is_skipped_entirely_when_no_server_is_configured(monkeypatch):
    class _Provider:
        configured = False

        async def load_tools(self):
            raise AssertionError("must not be called")

    monkeypatch.setattr(registry, "get_mcp_provider", lambda: _Provider())

    assert registry._load_mcp_tools() == []


def test_mcp_loading_failures_never_break_tool_building(monkeypatch, caplog):
    class _Provider:
        configured = True
        timeout = 1

        async def load_tools(self):
            raise ConnectionError("mcp server refused")

    monkeypatch.setattr(registry, "get_mcp_provider", lambda: _Provider())

    with caplog.at_level("ERROR", logger="src.tools.registry"):
        assert registry._load_mcp_tools() == []

    assert "MCP tool loading failed" in caplog.text


def test_mcp_tools_load_from_a_worker_thread_when_a_loop_is_already_running(monkeypatch):
    class _Provider:
        configured = True
        timeout = 1

        async def load_tools(self):
            return [_tool("external_lookup")]

    monkeypatch.setattr(registry, "get_mcp_provider", lambda: _Provider())

    async def _inside_a_loop():
        return registry._load_mcp_tools()

    tools = asyncio.run(_inside_a_loop())

    assert registry.tool_names(tools) == ["external_lookup"]


def _evidence(rows: int = 2) -> SqlEvidence:
    return SqlEvidence(
        template_id="vendor_status_by_name",
        statement="SELECT name, status FROM vendors WHERE name = %(name)s",
        parameters={"name": "Acme"},
        row_count=rows,
        rows=[{"name": "Acme", "status": "approved", "n": i} for i in range(rows)],
        as_of=settings.as_of_date,
    )


@pytest.fixture
def sql_runner(monkeypatch):
    calls = []

    def _install(outcome):
        def _run(**kwargs):
            calls.append(kwargs)

            if isinstance(outcome, Exception):
                raise outcome

            return outcome

        monkeypatch.setattr(nl2sql_tool, "run_vetted_sql", _run)
        monkeypatch.setattr(nl2sql_tool, "sanity_check", lambda evidence: ["as-of date pinned"])

        return calls

    return _install


def test_the_sql_tool_lists_only_the_templates_the_role_can_run():
    manager = nl2sql_tool.build_sql_tool({"vendors", "compliance_reviews"})
    officer = nl2sql_tool.build_sql_tool({"vendors", "compliance_reviews", "audit_logs", "retention_records"})

    assert manager.metadata.name == "compliance_records"
    assert "does not accept SQL" in manager.metadata.description
    assert len(officer.metadata.description) > len(manager.metadata.description)
    assert "retention" not in manager.metadata.description.lower().split("reviewed queries available")[1]


def test_the_sql_tool_returns_the_evidence_as_json_with_caveats(sql_runner):
    calls = sql_runner(_evidence())
    tool = nl2sql_tool.build_sql_tool({"vendors"}, departments=["Sales"], risk_categories={"Low"})

    payload = json.loads(tool.call("is Acme approved").raw_output)

    assert payload["template_id"] == "vendor_status_by_name"
    assert payload["row_count"] == 2
    assert payload["parameters"] == {"name": "Acme"}
    assert payload["as_of"] == str(settings.as_of_date)
    assert payload["caveats"] == ["as-of date pinned"]

    [call] = calls
    assert call["question"] == "is Acme approved"
    assert call["allowed_tables"] == {"vendors"}
    assert call["departments"] == ["Sales"]
    assert call["risk_categories"] == {"Low"}
    assert call["as_of"] == settings.as_of_date


def test_the_sql_tool_caps_the_rows_it_hands_to_the_agent(sql_runner):
    sql_runner(_evidence(rows=60))
    tool = nl2sql_tool.build_sql_tool({"vendors"})

    payload = json.loads(tool.call("list vendors").raw_output)

    assert payload["row_count"] == 60
    assert len(payload["rows"]) == 25


def test_a_policy_refusal_is_returned_to_the_agent_as_text(sql_runner, caplog):
    sql_runner(SqlPolicyError("table audit_logs is not granted to this role"))
    tool = nl2sql_tool.build_sql_tool({"vendors"})

    with caplog.at_level("WARNING", logger="src.tools.nl2sql_tool"):
        text = tool.call("show me the audit log").raw_output

    assert text.startswith("The database query was refused:")
    assert "audit_logs" in text
    assert "refused a call" in caplog.text


def test_an_unexpected_failure_is_returned_to_the_agent_not_raised(sql_runner):
    sql_runner(TimeoutError("statement timeout"))
    tool = nl2sql_tool.build_sql_tool({"vendors"})

    text = tool.call("list vendors").raw_output

    assert text.startswith("The database query could not be completed:")
    assert "statement timeout" in text


def test_a_role_with_no_tables_is_told_so_without_touching_the_database(sql_runner):
    calls = sql_runner(_evidence())
    tool = nl2sql_tool.build_sql_tool(set())

    assert tool.call("anything").raw_output == "This role has no access to the compliance database."
    assert calls == []


@pytest.fixture
def provider(monkeypatch):
    def _make(url: str = "", allowed: str = "", required: bool = False):
        monkeypatch.setattr(settings, "mcp_server_url", url)
        monkeypatch.setattr(settings, "mcp_allowed_tools", allowed)
        monkeypatch.setattr(settings, "mcp_required", required)
        monkeypatch.setattr(mcp_tools, "_provider", None)

        return mcp_tools.get_mcp_provider()

    return _make


def test_the_provider_is_unconfigured_without_a_url_and_loads_nothing(provider):
    p = provider()

    assert p.configured is False
    assert asyncio.run(p.load_tools()) == []


def test_the_allow_list_is_parsed_from_the_comma_separated_setting(provider):
    p = provider(url="http://mcp.local", allowed=" search , fetch,,")

    assert p.configured is True
    assert p.allowed == ["search", "fetch"]


def test_the_provider_is_a_singleton(provider):
    p = provider(url="http://mcp.local")

    assert mcp_tools.get_mcp_provider() is p


def test_a_failed_load_is_swallowed_unless_mcp_is_required(provider, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _no_mcp(name, *args, **kwargs):
        if name.startswith("llama_index.tools.mcp"):
            raise ImportError("llama-index-tools-mcp not installed")

        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_mcp)

    assert asyncio.run(provider(url="http://mcp.local").load_tools()) == []

    with pytest.raises(ConnectionError, match="required"):
        asyncio.run(provider(url="http://mcp.local", required=True).load_tools())


def test_the_clamp_truncates_long_tool_output_and_says_so(monkeypatch):
    class _Raw:
        def __init__(self, text):
            self.content = [type("Item", (), {"text": text})()]

    class _Result:
        def __init__(self, text):
            self.raw_output = _Raw(text)
            self.content = text

    class _Tool:
        metadata = _tool("remote").metadata

        async def acall(self, **kwargs):
            return _Result("x" * 100)

    clamped = mcp_tools._clamp(_Tool(), max_chars=40)

    text = asyncio.run(clamped.acall(input="q")).raw_output

    assert text.startswith("x" * 40)
    assert "truncated at 40 characters" in text
    assert clamped.metadata.name == "remote"


def test_the_clamp_leaves_short_output_untouched():
    class _Result:
        raw_output = None
        content = "short answer"

    class _Tool:
        metadata = _tool("remote").metadata

        async def acall(self, **kwargs):
            return _Result()

    clamped = mcp_tools._clamp(_Tool(), max_chars=40)

    assert asyncio.run(clamped.acall(input="q")).raw_output == "short answer"
