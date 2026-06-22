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
from xerxes.runtime.bridge import populate_registry
from xerxes.streaming.permissions import SAFE_TOOLS


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
    registry = populate_registry()
    schemas = {schema["name"]: schema for schema in registry.tool_schemas()}
    edit_props = schemas["FileEditTool"]["input_schema"]["properties"]
    shell_props = schemas["ExecuteShell"]["input_schema"]["properties"]

    assert edit_props["replace_all"]["type"] == "boolean"
    assert shell_props["timeout"]["type"] == "number"


def test_root_agent_tool_schemas_follow_default_agent_yaml() -> None:
    registry = populate_registry()
    default_agent = get_agent_definition("default")
    filtered = BridgeServer._filter_tool_schemas_for_agent(registry.tool_schemas(), default_agent)
    tool_names = {schema.get("name", "") for schema in filtered}

    assert len(filtered) < len(registry.tool_schemas())
    assert tool_names == set(default_agent.tools)
    assert "GoogleSearch" not in tool_names
    assert "WebScraper" not in tool_names
