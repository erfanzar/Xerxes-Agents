# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from xerxes.agents.definitions import get_agent_definition
from xerxes.bridge.server import BridgeServer
from xerxes.operators import OperatorRuntimeConfig, OperatorState
from xerxes.runtime.bridge import create_query_engine, populate_registry, register_operator_tools
from xerxes.streaming.permissions import SAFE_TOOLS

TERMINAL_OPERATOR_TOOLS = {
    "exec_command",
    "write_stdin",
    "list_terminal_sessions",
    "close_terminal_session",
}


def _registry_with_terminal_tools():
    registry = populate_registry()
    operator_state = OperatorState(
        OperatorRuntimeConfig(
            enabled=True,
            power_tools_enabled=True,
            allowed_tool_names=set(TERMINAL_OPERATOR_TOOLS),
        )
    )
    return register_operator_tools(registry, operator_state, set(TERMINAL_OPERATOR_TOOLS))


def test_agent_registry_excludes_google_search_and_web_scraper() -> None:
    registry = populate_registry()
    tool_names = {schema.get("name", "") for schema in registry.tool_schemas()}

    assert "GoogleSearch" not in tool_names
    assert "WebScraper" not in tool_names
    assert "DuckDuckGoSearch" in tool_names


def test_safe_tool_allowlist_contains_only_registered_tools() -> None:
    registry = populate_registry()
    tool_names = {schema.get("name", "") for schema in registry.tool_schemas()}

    assert "GoogleSearch" not in SAFE_TOOLS
    assert "WebScraper" not in SAFE_TOOLS
    assert SAFE_TOOLS <= tool_names


def test_registry_prefers_explicit_read_file_schema() -> None:
    registry = populate_registry()
    schemas = {schema["name"]: schema for schema in registry.tool_schemas()}
    read_props = schemas["ReadFile"]["input_schema"]["properties"]

    assert read_props["offset"]["type"] == "integer"
    assert read_props["limit"]["type"] == "integer"
    assert read_props["limit"]["default"] == 400
    assert "Pass -1" in read_props["limit"]["description"]


def test_registry_reflects_bool_and_numeric_types() -> None:
    registry = _registry_with_terminal_tools()
    schemas = {schema["name"]: schema for schema in registry.tool_schemas()}
    edit_props = schemas["FileEditTool"]["input_schema"]["properties"]
    exec_props = schemas["exec_command"]["input_schema"]["properties"]

    assert edit_props["replace_all"]["type"] == "boolean"
    assert exec_props["yield_time_ms"]["type"] == "integer"


def test_registry_no_longer_exposes_blocking_execute_shell() -> None:
    registry = populate_registry()
    tool_names = {schema.get("name", "") for schema in registry.tool_schemas()}

    removed_shell_tool = "Execute" + "Shell"
    assert removed_shell_tool not in tool_names


def test_root_agent_tool_schemas_follow_default_agent_yaml() -> None:
    registry = _registry_with_terminal_tools()
    default_agent = get_agent_definition("default")
    filtered = BridgeServer._filter_tool_schemas_for_agent(registry.tool_schemas(), default_agent)
    tool_names = {schema.get("name", "") for schema in filtered}

    assert len(filtered) < len(registry.tool_schemas())
    assert tool_names == set(default_agent.tools)
    assert "GoogleSearch" not in tool_names
    assert "WebScraper" not in tool_names


def test_code_agents_can_use_terminal_sessions() -> None:
    default_agent = get_agent_definition("default")
    coder_agent = get_agent_definition("coder")

    assert default_agent is not None
    assert coder_agent is not None
    assert TERMINAL_OPERATOR_TOOLS <= set(default_agent.tools)
    assert TERMINAL_OPERATOR_TOOLS <= set(coder_agent.allowed_tools or [])


def test_create_query_engine_registers_terminal_operator_tools() -> None:
    engine = create_query_engine(model="test-model")
    schemas = engine._default_tool_schemas
    tool_names = {schema["name"] for schema in schemas}

    assert TERMINAL_OPERATOR_TOOLS <= tool_names
