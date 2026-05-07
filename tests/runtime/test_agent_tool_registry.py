from __future__ import annotations

from xerxes.agents.definitions import get_agent_definition
from xerxes.bridge.server import BridgeServer
from xerxes.runtime.bridge import populate_registry


def test_agent_registry_excludes_google_search_and_web_scraper() -> None:
    registry = populate_registry()
    tool_names = {schema.get("name", "") for schema in registry.tool_schemas()}

    assert "GoogleSearch" not in tool_names
    assert "WebScraper" not in tool_names
    assert "DuckDuckGoSearch" in tool_names


def test_root_agent_tool_schemas_follow_default_agent_yaml() -> None:
    registry = populate_registry()
    default_agent = get_agent_definition("default")
    filtered = BridgeServer._filter_tool_schemas_for_agent(registry.tool_schemas(), default_agent)
    tool_names = {schema.get("name", "") for schema in filtered}

    assert len(filtered) < len(registry.tool_schemas())
    assert tool_names == set(default_agent.tools)
    assert "GoogleSearch" not in tool_names
    assert "WebScraper" not in tool_names
