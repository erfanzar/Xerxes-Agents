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

from pathlib import Path

import pytest
from PIL import Image
from xerxes import Agent, OperatorRuntimeConfig, RuntimeFeaturesConfig, Xerxes
from xerxes.core.utils import function_to_json
from xerxes.tools.google_search import GoogleSearch
from xerxes.types import ExecutionStatus, ImageChunk, RequestFunctionCall, TextChunk, UserMessage


def test_operator_tools_use_public_names_and_dotted_aliases():
    xerxes = Xerxes(
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            operator=OperatorRuntimeConfig(enabled=True),
        )
    )
    agent = Agent(id="operator", model="fake", instructions="Use operator tools.", functions=[])
    xerxes.register_agent(agent)

    mapping = agent.get_functions_mapping()
    assert "web.time" in mapping
    assert "exec_command" in mapping

    schema = function_to_json(mapping["web.time"])
    assert schema["function"]["name"] == "web.time"


def test_operator_tool_schema_descriptions_are_detailed():
    xerxes = Xerxes(
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            operator=OperatorRuntimeConfig(enabled=True),
        )
    )
    agent = Agent(id="operator", model="fake", instructions="Use operator tools.", functions=[])
    xerxes.register_agent(agent)

    mapping = agent.get_functions_mapping()
    exec_schema = function_to_json(mapping["exec_command"])
    exec_props = exec_schema["function"]["parameters"]["properties"]
    assert "PTY-backed shell session" in exec_schema["function"]["description"]
    assert "interactive terminal session" in exec_schema["function"]["description"]
    assert exec_props["cmd"]["type"] == "string"
    assert "Shell command to launch" in exec_props["cmd"]["description"]

    search_schema = function_to_json(mapping["web.search_query"])
    search_props = search_schema["function"]["parameters"]["properties"]
    assert "Google" in search_schema["function"]["description"]
    assert "up-to-date information" in search_schema["function"]["description"]
    assert "domains" in search_props


def test_web_search_routes_through_google_search(monkeypatch):
    """``web.search_query`` operator wraps GoogleSearch.static_call."""
    captured: dict = {}

    def fake_static(query, n_results=5, site=None, time_range=None, **kw):
        captured.update(query=query, n_results=n_results, site=site, time_range=time_range)
        return {
            "engine": "google_api",
            "query": query,
            "count": 1,
            "results": [{"title": "OpenAI shipping news", "url": "https://example.com/openai-news", "snippet": "x"}],
        }

    monkeypatch.setattr(GoogleSearch, "static_call", staticmethod(fake_static))

    from xerxes.operators.config import OperatorRuntimeConfig
    from xerxes.operators.state import OperatorState

    state = OperatorState(OperatorRuntimeConfig(enabled=True, power_tools_enabled=True))
    web_search = state._build_web_search_query()
    payload = web_search("latest OpenAI news", search_type="news", n_results=3, domains=["openai.com"])

    assert payload["results"][0]["title"] == "OpenAI shipping news"
    assert payload["query"] == "latest OpenAI news"
    assert captured["site"] == "openai.com"
    assert captured["time_range"] == "d"
    assert captured["n_results"] == 3


@pytest.mark.asyncio
async def test_operator_policy_defaults_allow_power_tools():
    xerxes = Xerxes(
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            operator=OperatorRuntimeConfig(enabled=True),
        )
    )
    agent = Agent(id="operator", model="fake", instructions="Use operator tools.", functions=[])
    xerxes.register_agent(agent)

    safe_call = await xerxes.executor._execute_single_call(
        RequestFunctionCall(name="web.time", arguments={"utc_offset": "+03:00"}),
        {},
        agent,
        runtime_features_state=xerxes._runtime_features_state,
    )
    power_call = await xerxes.executor._execute_single_call(
        RequestFunctionCall(name="exec_command", arguments={"cmd": "printf hi"}),
        {},
        agent,
        runtime_features_state=xerxes._runtime_features_state,
    )

    assert safe_call.status == ExecutionStatus.SUCCESS
    assert power_call.status == ExecutionStatus.SUCCESS


def test_view_image_creates_multimodal_reinvoke_message(tmp_path: Path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (16, 12), color="navy").save(image_path)

    xerxes = Xerxes(
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            operator=OperatorRuntimeConfig(enabled=True, power_tools_enabled=True),
        )
    )
    agent = Agent(id="operator", model="fake", instructions="Use operator tools.", functions=[])
    xerxes.register_agent(agent)

    operator_state = xerxes._runtime_features_state.operator_state
    assert operator_state is not None
    view_image = agent.get_functions_mapping()["view_image"]
    result = view_image(path=str(image_path))
    message = operator_state.create_reinvoke_message(result)

    assert isinstance(message, UserMessage)
    assert isinstance(message.content[0], TextChunk)
    assert "[TOOL IMAGE RESULT]" in message.content[0].text
    assert isinstance(message.content[1], ImageChunk)


def test_update_plan_mutates_operator_state():
    xerxes = Xerxes(
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            operator=OperatorRuntimeConfig(enabled=True),
        )
    )
    agent = Agent(id="operator", model="fake", instructions="Use operator tools.", functions=[])
    xerxes.register_agent(agent)

    update_plan = agent.get_functions_mapping()["update_plan"]
    payload = update_plan(
        explanation="Track execution.",
        plan=[{"step": "Implement tools", "status": "in_progress"}],
    )

    assert payload["revision"] == 1
    assert payload["steps"][0]["step"] == "Implement tools"
