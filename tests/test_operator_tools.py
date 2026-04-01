# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from calute import Agent, Calute, OperatorRuntimeConfig, RuntimeFeaturesConfig
from calute.core.utils import function_to_json
from calute.tools.duckduckgo_engine import DuckDuckGoSearch
from calute.types import ExecutionStatus, ImageChunk, RequestFunctionCall, TextChunk, UserMessage


def test_operator_tools_use_public_names_and_dotted_aliases():
    calute = Calute(
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            operator=OperatorRuntimeConfig(enabled=True),
        )
    )
    agent = Agent(id="operator", model="fake", instructions="Use operator tools.", functions=[])
    calute.register_agent(agent)

    mapping = agent.get_functions_mapping()
    assert "web.time" in mapping
    assert "exec_command" in mapping

    schema = function_to_json(mapping["web.time"])
    assert schema["function"]["name"] == "web.time"


def test_operator_tool_schema_descriptions_are_detailed():
    calute = Calute(
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            operator=OperatorRuntimeConfig(enabled=True),
        )
    )
    agent = Agent(id="operator", model="fake", instructions="Use operator tools.", functions=[])
    calute.register_agent(agent)

    mapping = agent.get_functions_mapping()
    exec_schema = function_to_json(mapping["exec_command"])
    exec_props = exec_schema["function"]["parameters"]["properties"]
    assert "PTY-backed shell session" in exec_schema["function"]["description"]
    assert "interactive terminal session" in exec_schema["function"]["description"]
    assert exec_props["cmd"]["type"] == "string"
    assert "Shell command to launch" in exec_props["cmd"]["description"]

    search_schema = function_to_json(mapping["web.search_query"])
    search_props = search_schema["function"]["parameters"]["properties"]
    assert "DuckDuckGo" in search_schema["function"]["description"]
    assert "up-to-date information" in search_schema["function"]["description"]
    assert "domain allowlist" in search_props["domains"]["description"]


def test_duckduckgo_news_search_falls_back_to_text(monkeypatch):
    class _FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def news(self, *args, **kwargs):
            raise RuntimeError("No results found for query")

        def text(self, *args, **kwargs):
            return [
                {
                    "title": "OpenAI shipping news",
                    "href": "https://example.com/openai-news",
                    "body": "Fallback text result.",
                }
            ]

    monkeypatch.setattr("calute.tools.duckduckgo_engine._get_ddgs", lambda: _FakeDDGS)

    payload = DuckDuckGoSearch.static_call(
        "latest OpenAI news",
        search_type="news",
        n_results=3,
        return_metadata=True,
    )

    assert payload["results"][0]["title"] == "OpenAI shipping news"
    assert payload["metadata"]["effective_search_type"] == "text"
    assert payload["metadata"]["fallback_applied"] == "news_to_text"


@pytest.mark.asyncio
async def test_operator_policy_defaults_allow_power_tools():
    calute = Calute(
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            operator=OperatorRuntimeConfig(enabled=True),
        )
    )
    agent = Agent(id="operator", model="fake", instructions="Use operator tools.", functions=[])
    calute.register_agent(agent)

    safe_call = await calute.executor._execute_single_call(
        RequestFunctionCall(name="web.time", arguments={"utc_offset": "+03:00"}),
        {},
        agent,
        runtime_features_state=calute._runtime_features_state,
    )
    power_call = await calute.executor._execute_single_call(
        RequestFunctionCall(name="exec_command", arguments={"cmd": "printf hi"}),
        {},
        agent,
        runtime_features_state=calute._runtime_features_state,
    )

    assert safe_call.status == ExecutionStatus.SUCCESS
    assert power_call.status == ExecutionStatus.SUCCESS


def test_view_image_creates_multimodal_reinvoke_message(tmp_path: Path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (16, 12), color="navy").save(image_path)

    calute = Calute(
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            operator=OperatorRuntimeConfig(enabled=True, power_tools_enabled=True),
        )
    )
    agent = Agent(id="operator", model="fake", instructions="Use operator tools.", functions=[])
    calute.register_agent(agent)

    operator_state = calute._runtime_features_state.operator_state
    assert operator_state is not None
    view_image = agent.get_functions_mapping()["view_image"]
    result = view_image(path=str(image_path))
    message = operator_state.create_reinvoke_message(result)

    assert isinstance(message, UserMessage)
    assert isinstance(message.content[0], TextChunk)
    assert "[TOOL IMAGE RESULT]" in message.content[0].text
    assert isinstance(message.content[1], ImageChunk)


def test_update_plan_mutates_operator_state():
    calute = Calute(
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            operator=OperatorRuntimeConfig(enabled=True),
        )
    )
    agent = Agent(id="operator", model="fake", instructions="Use operator tools.", functions=[])
    calute.register_agent(agent)

    update_plan = agent.get_functions_mapping()["update_plan"]
    payload = update_plan(
        explanation="Track execution.",
        plan=[{"step": "Implement tools", "status": "in_progress"}],
    )

    assert payload["revision"] == 1
    assert payload["steps"][0]["step"] == "Implement tools"
