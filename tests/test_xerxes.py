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
"""Comprehensive tests for xerxes.xerxes — the main Xerxes orchestration class."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest
from xerxes import Agent, Xerxes
from xerxes.core.prompt_template import PromptSection, PromptTemplate
from xerxes.core.streamer_buffer import StreamerBuffer
from xerxes.llms.base import BaseLLM, LLMConfig
from xerxes.memory import MemoryStore
from xerxes.runtime.features import RuntimeFeaturesConfig
from xerxes.types import (
    AgentSwitch,
    Completion,
    ExecutionStatus,
    FunctionCallsExtracted,
    FunctionDetection,
    FunctionExecutionComplete,
    FunctionExecutionStart,
    ReinvokeSignal,
    RequestFunctionCall,
    ResponseResult,
    StreamChunk,
    ToolCall,
)
from xerxes.types.messages import (
    AssistantMessage,
    MessagesHistory,
    SystemMessage,
    TextChunk,
    ToolMessage,
    UserMessage,
)
from xerxes.types.tool_calls import FunctionCall


def _chunk(
    *,
    content: str | None = None,
    function_calls: list[dict] | None = None,
    is_final: bool = False,
    streaming_tool_calls: dict | None = None,
) -> dict:
    buffered_content = content or ""
    return {
        "content": content,
        "buffered_content": buffered_content,
        "reasoning_content": None,
        "buffered_reasoning_content": "",
        "function_calls": function_calls or [],
        "tool_calls": None,
        "streaming_tool_calls": streaming_tool_calls,
        "raw_chunk": None,
        "is_final": is_final,
    }


class _FakeLLM(BaseLLM):
    """Fake LLM that yields pre-canned chunks for testing."""

    def __init__(self, responses: list[list[dict]]):
        self.responses = list(responses)
        self.calls: list[dict] = []
        super().__init__(config=LLMConfig(model="fake-model"))

    def _initialize_client(self) -> None:
        self.client = object()

    async def generate_completion(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return self.responses.pop(0)

    def extract_content(self, response) -> str:
        return ""

    async def process_streaming_response(self, response, callback):
        output = ""
        for chunk in response:
            content = chunk.get("content")
            if content:
                callback(content, chunk)
                output += content
        return output

    def stream_completion(self, response, agent=None):
        yield from response

    async def astream_completion(self, response, agent=None):
        for chunk in response:
            yield chunk


class _FakeLLMAsyncResponse(_FakeLLM):
    """Fake LLM whose generate_completion returns an async-iterable response."""

    async def generate_completion(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        chunks = self.responses.pop(0)

        class AsyncIterable:
            def __init__(self, items):
                self.items = items

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.items:
                    raise StopAsyncIteration
                return self.items.pop(0)

        return AsyncIterable(list(chunks))


def _make_tool_fn(name: str = "test_tool"):
    def tool_fn(query: str) -> str:
        """A test tool.

        Args:
            query: The query string.
        """
        return f"result:{query}"

    tool_fn.__name__ = name
    return tool_fn


class TestXerxesInit:
    def test_init_defaults(self):
        xerxes = Xerxes()
        assert xerxes.llm_client is None
        assert isinstance(xerxes.template, PromptTemplate)
        assert xerxes.enable_memory is False
        assert xerxes.auto_add_memory_tools is True
        assert xerxes._runtime_features_state is not None
        assert xerxes.runtime_features.enabled is True

    def test_init_with_llm(self):
        llm = _FakeLLM(responses=[])
        xerxes = Xerxes(llm=llm)
        assert xerxes.llm_client is llm

    def test_init_with_custom_template(self):
        template = PromptTemplate(sections={PromptSection.SYSTEM: "CUSTOM:"})
        xerxes = Xerxes(template=template)
        assert xerxes.template is template

    def test_init_with_memory(self):
        xerxes = Xerxes(enable_memory=True, memory_config={"max_short_term": 50})
        assert xerxes.enable_memory is True
        assert isinstance(xerxes.memory_store, MemoryStore)

    def test_init_memory_disabled_no_store(self):
        xerxes = Xerxes(enable_memory=False)
        assert not hasattr(xerxes, "memory_store")

    def test_init_runtime_features_none_gets_defaults(self):
        xerxes = Xerxes(runtime_features=None)
        assert xerxes.runtime_features.enabled is True
        assert xerxes.runtime_features.operator is not None
        assert xerxes.runtime_features.operator.enabled is True

    def test_init_runtime_features_with_workspace_root(self):
        rf = RuntimeFeaturesConfig(enabled=True, workspace_root="/tmp/test")
        xerxes = Xerxes(runtime_features=rf)
        assert xerxes.runtime_features.workspace_root == "/tmp/test"

    def test_init_runtime_features_enabled_no_operator(self):
        rf = RuntimeFeaturesConfig(enabled=True, operator=None)
        xerxes = Xerxes(runtime_features=rf)
        assert xerxes.runtime_features.operator is not None
        assert xerxes.runtime_features.operator.enabled is True

    def test_init_runtime_features_operator_no_workdir(self):
        from xerxes.operators import OperatorRuntimeConfig

        rf = RuntimeFeaturesConfig(
            enabled=True,
            operator=OperatorRuntimeConfig(enabled=True, shell_default_workdir=None, power_tools_enabled=True),
        )
        xerxes = Xerxes(runtime_features=rf)
        assert xerxes.runtime_features.operator.shell_default_workdir is not None

    def test_init_runtime_disabled(self):
        rf = RuntimeFeaturesConfig(enabled=False)
        xerxes = Xerxes(runtime_features=rf)
        assert xerxes._runtime_features_state is None
        assert xerxes._session_id is None


class TestXerxesRuntimeFeatures:
    def test_normalize_runtime_features_none(self):
        result = Xerxes._normalize_runtime_features(None, "/workspace")
        assert result.enabled is True
        assert result.workspace_root == "/workspace"
        assert result.operator is not None
        assert result.operator.power_tools_enabled is True

    def test_normalize_runtime_features_preserves_existing(self):
        rf = RuntimeFeaturesConfig(enabled=True, workspace_root="/ existing")
        result = Xerxes._normalize_runtime_features(rf, "/workspace")
        assert result.workspace_root == "/ existing"

    def test_setup_default_triggers_registers_two(self):
        xerxes = Xerxes()
        from xerxes.types import AgentSwitchTrigger

        assert AgentSwitchTrigger.CAPABILITY_BASED in xerxes.orchestrator.switch_triggers
        assert AgentSwitchTrigger.ERROR_RECOVERY in xerxes.orchestrator.switch_triggers


class TestXerxesAgentManagement:
    def test_register_agent_sets_current(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        assert xerxes.orchestrator.current_agent_id == "a1"

    def test_register_agent_adds_memory_tools_when_enabled(self):
        xerxes = Xerxes(enable_memory=True, auto_add_memory_tools=True)
        agent = Agent(id="a1", model="fake", functions=[])
        xerxes.register_agent(agent)
        func_names = agent.get_available_functions()
        assert any("memory" in fn.lower() for fn in func_names)

    def test_register_agent_skips_memory_tools_when_disabled(self):
        xerxes = Xerxes(enable_memory=False)
        agent = Agent(id="a1", model="fake", functions=[])
        xerxes.register_agent(agent)
        func_names = agent.get_available_functions()
        assert not any("memory" in fn.lower() for fn in func_names)

    def test_register_agent_with_runtime_features_merges_tools(self):
        xerxes = Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=True))
        agent = Agent(id="a1", model="fake", functions=[])
        xerxes.register_agent(agent)

        assert xerxes.orchestrator.agents["a1"] is agent

    def test_register_multiple_agents(self):
        xerxes = Xerxes()
        a1 = Agent(id="a1", model="fake")
        a2 = Agent(id="a2", model="fake")
        xerxes.register_agent(a1)
        xerxes.register_agent(a2)
        assert xerxes.orchestrator.current_agent_id == "a1"
        assert len(xerxes.orchestrator.agents) == 2


class TestXerxesMessageManagement:
    def test_manage_messages_no_agent_returns_user_message(self):
        xerxes = Xerxes()
        messages = xerxes.manage_messages(agent=None, prompt="hello")
        assert len(messages.messages) == 1
        assert isinstance(messages.messages[0], UserMessage)
        assert messages.messages[0].content == "hello"

    def test_manage_messages_builds_system_prompt(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="fake", instructions="Be helpful.")
        messages = xerxes.manage_messages(agent=agent, prompt="hello")
        assert isinstance(messages.messages[0], SystemMessage)
        assert "Be helpful" in messages.messages[0].content

    def test_manage_messages_with_chain_of_thought(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="fake", instructions="Be helpful.")
        messages = xerxes.manage_messages(agent=agent, prompt="hello", use_chain_of_thought=True)
        system = messages.messages[0].content
        assert "systematically" in system

    def test_manage_messages_with_instructed_prompt(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="fake", instructions="Be helpful.", functions=[_make_tool_fn()])
        messages = xerxes.manage_messages(agent=agent, prompt="hello", use_instructed_prompt=True)
        system = messages.messages[0].content
        assert "SYSTEM" in system or "system" in system.lower()

    def test_manage_messages_with_memory(self):
        from xerxes.memory import MemoryType

        xerxes = Xerxes(enable_memory=True)
        agent = Agent(id="a1", model="fake", instructions="Be helpful.")
        xerxes.register_agent(agent)
        xerxes.memory_store.add_memory("past context", MemoryType.SHORT_TERM, agent_id="a1")
        messages = xerxes.manage_messages(agent=agent, prompt="hello", include_memory=True)
        system = messages.messages[0].content
        assert "past context" in system or "Relevant information" in system

    def test_manage_messages_with_context_variables(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="fake", instructions="Be helpful.")
        messages = xerxes.manage_messages(agent=agent, prompt="hello", context_variables={"key": "value"})
        system = messages.messages[0].content
        assert "value" in system

    def test_manage_messages_with_require_reflection(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="fake", instructions="Be helpful.")
        messages = xerxes.manage_messages(agent=agent, prompt="hello", require_reflection=True)
        user_msg = messages.messages[-1]
        assert isinstance(user_msg, UserMessage)
        assert "reflection" in user_msg.content

    def test_manage_messages_with_history(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="fake", instructions="Be helpful.")
        history = MessagesHistory(messages=[UserMessage(content="prior")])
        messages = xerxes.manage_messages(agent=agent, prompt="hello", messages=history)
        assert len(messages.messages) == 3

    def test_manage_messages_with_callable_instructions(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="fake", instructions=lambda: "Dynamic.")
        messages = xerxes.manage_messages(agent=agent, prompt="hello")
        assert "Dynamic." in messages.messages[0].content

    def test_manage_messages_with_callable_rules(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="fake", instructions="Be helpful.", rules=lambda: ["Rule A"])
        messages = xerxes.manage_messages(agent=agent, prompt="hello")
        assert "Rule A" in messages.messages[0].content

    def test_manage_messages_with_examples(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="fake", instructions="Be helpful.", examples=["Ex1", "Ex2"])
        messages = xerxes.manage_messages(agent=agent, prompt="hello")
        assert "Ex1" in messages.messages[0].content
        assert "Ex2" in messages.messages[0].content

    def test_merge_system_history_deduplicates(self):
        xerxes = Xerxes()
        history = MessagesHistory(
            messages=[
                SystemMessage(content="sys1"),
                UserMessage(content="user1"),
                SystemMessage(content="sys1"),
            ]
        )
        merged, remaining = xerxes._merge_system_history("base", history)
        assert "base" in merged
        assert "sys1" in merged
        assert len(remaining) == 1
        assert isinstance(remaining[0], UserMessage)

    def test_system_message_to_text_with_string(self):
        msg = SystemMessage(content="hello")
        assert Xerxes._system_message_to_text(msg) == "hello"

    def test_system_message_to_text_with_empty_string(self):
        msg = SystemMessage(content="   ")
        assert Xerxes._system_message_to_text(msg) is None

    def test_format_section_with_list(self):
        xerxes = Xerxes()
        result = xerxes._format_section("HEADER:", ["a", "b"])
        assert result is not None
        assert "HEADER:" in result
        assert "- a" in result

    def test_format_section_with_string(self):
        xerxes = Xerxes()
        result = xerxes._format_section("HEADER:", "content")
        assert "content" in result

    def test_format_section_empty_returns_none(self):
        xerxes = Xerxes()
        assert xerxes._format_section("HEADER:", None) is None
        assert xerxes._format_section("HEADER:", "") is None
        assert xerxes._format_section("HEADER:", []) is None

    def test_format_section_no_header(self):
        xerxes = Xerxes()
        result = xerxes._format_section("", "content")
        assert result == "content"


class TestXerxesFunctionExtraction:
    def test_extract_from_markdown_found_json(self):
        content = '```json\n{"key": "value"}\n```'
        result = Xerxes.extract_from_markdown("json", content)
        assert result == {"key": "value"}

    def test_extract_from_markdown_found_string(self):
        content = "```python\nprint(1)\n```"
        result = Xerxes.extract_from_markdown("python", content)
        assert result == "print(1)"

    def test_extract_from_markdown_not_found(self):
        assert Xerxes.extract_from_markdown("yaml", "no code") is None

    def test_extract_md_block(self):
        text = "```python\nprint(1)\n```\n```xml\n<a/>\n```"
        result = Xerxes.extract_md_block(text)
        assert ("python", "print(1)") in result
        assert ("xml", "<a/>") in result

    def test_extract_md_block_empty_lang(self):
        text = "```\nplain\n```"
        result = Xerxes.extract_md_block(text)
        assert ("", "plain") in result

    def test_detect_function_calls_no_functions(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="fake", functions=[])
        assert xerxes._detect_function_calls("<tool>", agent) is False

    def test_detect_function_calls_xml(self):
        xerxes = Xerxes()
        tool = _make_tool_fn("my_tool")
        agent = Agent(id="a1", model="fake", functions=[tool])
        content = '<my_tool><arguments>{"x":1}</arguments></my_tool>'
        assert xerxes._detect_function_calls(content, agent) is True

    def test_detect_function_calls_tagged(self):
        xerxes = Xerxes()
        tool = _make_tool_fn("my_tool")
        agent = Agent(id="a1", model="fake", functions=[tool])
        content = "<function=my_tool><parameter=x>1</parameter></function>"
        assert xerxes._detect_function_calls(content, agent) is True

    def test_detect_function_calls_tool_call_markdown(self):
        xerxes = Xerxes()
        tool = _make_tool_fn("my_tool")
        agent = Agent(id="a1", model="fake", functions=[tool])
        assert xerxes._detect_function_calls("```tool_call\n{}", agent) is True

    def test_detect_function_calls_regex_xml(self):
        xerxes = Xerxes()
        tool = _make_tool_fn("my_tool")
        agent = Agent(id="a1", model="fake", functions=[tool])
        content = '<my_tool><arguments>{"x":1}</arguments></my_tool>'
        assert xerxes._detect_function_calls_regex(content, agent) is True

    def test_detect_function_calls_regex_tagged(self):
        xerxes = Xerxes()
        tool = _make_tool_fn("my_tool")
        agent = Agent(id="a1", model="fake", functions=[tool])
        content = "<function=my_tool><parameter=x>1</parameter></function>"
        assert xerxes._detect_function_calls_regex(content, agent) is True

    def test_extract_function_calls_from_xml(self):
        xerxes = Xerxes()
        tool = _make_tool_fn("my_tool")
        agent = Agent(id="a1", model="fake", functions=[tool])
        content = '<my_tool><arguments>{"query":"hello"}</arguments></my_tool>'
        calls = xerxes._extract_function_calls_from_xml(content, agent)
        assert len(calls) == 1
        assert calls[0].name == "my_tool"
        assert calls[0].arguments == {"query": "hello"}

    def test_extract_function_calls_from_xml_ignores_unknown(self):
        xerxes = Xerxes()
        tool = _make_tool_fn("my_tool")
        agent = Agent(id="a1", model="fake", functions=[tool])
        content = '<unknown><arguments>{"x":1}</arguments></unknown>'
        calls = xerxes._extract_function_calls_from_xml(content, agent)
        assert len(calls) == 0

    def test_extract_function_calls_from_xml_bad_json(self):
        xerxes = Xerxes()
        tool = _make_tool_fn("my_tool")
        agent = Agent(id="a1", model="fake", functions=[tool])
        content = "<my_tool><arguments>not json</arguments></my_tool>"
        calls = xerxes._extract_function_calls_from_xml(content, agent)
        assert len(calls) == 0

    def test_extract_function_calls_from_tagged_markup(self):
        xerxes = Xerxes()
        tool = _make_tool_fn("my_tool")
        agent = Agent(id="a1", model="fake", functions=[tool])
        content = "<function=my_tool><parameter=query>hello</parameter></function>"
        calls = xerxes._extract_function_calls_from_tagged_markup(content, agent)
        assert len(calls) == 1
        assert calls[0].name == "my_tool"

    def test_extract_function_calls_from_tagged_markup_json_value(self):
        xerxes = Xerxes()
        tool = _make_tool_fn("my_tool")
        agent = Agent(id="a1", model="fake", functions=[tool])
        content = '<function=my_tool><parameter=query>{"a":1}</parameter></function>'
        calls = xerxes._extract_function_calls_from_tagged_markup(content, agent)
        assert len(calls) == 1
        assert calls[0].arguments == {"query": {"a": 1}}

    def test_extract_function_calls_from_tagged_markup_required_missing(self):
        xerxes = Xerxes()
        tool = _make_tool_fn("my_tool")
        agent = Agent(id="a1", model="fake", functions=[tool])

        content = "<function=my_tool></function>"
        calls = xerxes._extract_function_calls_from_tagged_markup(content, agent)
        assert len(calls) == 0

    def test_extract_function_calls_from_markdown(self):
        xerxes = Xerxes()
        tool = _make_tool_fn("my_tool")
        agent = Agent(id="a1", model="fake", functions=[tool])
        content = '```tool_call\n{"name": "my_tool", "content": {"query": "hi"}}\n```'
        calls = xerxes._extract_function_calls(content, agent)
        assert len(calls) == 1
        assert calls[0].name == "my_tool"

    def test_extract_function_calls_with_tool_calls(self):
        xerxes = Xerxes()
        tool = _make_tool_fn("my_tool")
        agent = Agent(id="a1", model="fake", functions=[tool])
        tc = ToolCall(
            id="t1",
            function=FunctionCall(name="my_tool", arguments='{"query":"hi"}'),
        )
        calls = xerxes._extract_function_calls("", agent, tool_calls=[tc])
        assert len(calls) == 1
        assert calls[0].arguments == {"query": "hi"}

    def test_extract_function_calls_with_tool_calls_string_arguments(self):
        xerxes = Xerxes()
        tool = _make_tool_fn("my_tool")
        agent = Agent(id="a1", model="fake", functions=[tool])
        tc = ToolCall(
            id="t1",
            function=FunctionCall(name="my_tool", arguments='{"query":"hi"}'),
        )
        calls = xerxes._extract_function_calls("", agent, tool_calls=[tc])
        assert calls[0].arguments == {"query": "hi"}

    def test_extract_function_calls_with_tool_calls_bad_json(self):
        xerxes = Xerxes()
        tool = _make_tool_fn("my_tool")
        agent = Agent(id="a1", model="fake", functions=[tool])
        # Bypass Pydantic validation to simulate malformed JSON from a provider
        fc = FunctionCall.model_construct(name="my_tool", arguments='{"broken"')
        tc = ToolCall.model_construct(id="t1", function=fc)
        calls = xerxes._extract_function_calls("", agent, tool_calls=[tc])

        assert len(calls) == 1

    def test_convert_function_calls(self):
        xerxes = Xerxes()
        tool = _make_tool_fn("my_tool")
        agent = Agent(id="a1", model="fake", functions=[tool])
        data = [{"name": "my_tool", "arguments": {"query": "hi"}, "id": "c1"}]
        calls = xerxes._convert_function_calls(data, agent)
        assert len(calls) == 1
        assert calls[0].id == "c1"

    def test_convert_function_calls_ignores_unknown(self):
        xerxes = Xerxes()
        tool = _make_tool_fn("my_tool")
        agent = Agent(id="a1", model="fake", functions=[tool])
        data = [{"name": "unknown", "arguments": {}}]
        calls = xerxes._convert_function_calls(data, agent)
        assert len(calls) == 0

    def test_convert_function_calls_string_arguments(self):
        xerxes = Xerxes()
        tool = _make_tool_fn("my_tool")
        agent = Agent(id="a1", model="fake", functions=[tool])
        data = [{"name": "my_tool", "arguments": '{"query":"hi"}', "id": "c1"}]
        calls = xerxes._convert_function_calls(data, agent)
        assert calls[0].arguments == {"query": "hi"}

    def test_remove_function_calls_from_content(self):
        xerxes = Xerxes()
        content = "Hello <func><arguments>{}</arguments></func> world"
        cleaned = xerxes._remove_function_calls_from_content(content)
        assert "Hello" in cleaned
        assert "world" in cleaned
        assert "<func>" not in cleaned

    def test_remove_function_calls_tagged(self):
        xerxes = Xerxes()
        content = "Hello <function=fn>args</function> world"
        cleaned = xerxes._remove_function_calls_from_content(content)
        assert "<function=" not in cleaned

    def test_remove_function_calls_markdown(self):
        xerxes = Xerxes()
        content = "Hello ```tool_call\n{}\n``` world"
        cleaned = xerxes._remove_function_calls_from_content(content)
        assert "tool_call" not in cleaned


class TestXerxesUtilityMethods:
    def test_get_thoughts_found(self):
        assert Xerxes.get_thoughts("a <think>b</think> c") == "b"

    def test_get_thoughts_not_found(self):
        assert Xerxes.get_thoughts("no thoughts") is None

    def test_get_thoughts_custom_tag(self):
        assert Xerxes.get_thoughts("<reason>r</reason>", tag="reason") == "r"

    def test_filter_thoughts(self):
        assert Xerxes.filter_thoughts("a <think>b</think> c") == "a  c"

    def test_filter_thoughts_custom_tag(self):
        assert Xerxes.filter_thoughts("a <reason>b</reason> c", tag="reason") == "a  c"

    def test_format_function_parameters(self):
        xerxes = Xerxes()
        params = {
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "context_variables": {"type": "object"},
            },
            "required": ["query"],
        }
        result = xerxes.format_function_parameters(params)
        assert "query" in result
        assert "required" in result
        assert "context_variables" not in result

    def test_format_function_parameters_empty(self):
        xerxes = Xerxes()
        assert xerxes.format_function_parameters({}) == ""

    def test_format_context_variables(self):
        xerxes = Xerxes()
        result = xerxes.format_context_variables({"name": "test", "count": 5})
        assert "name (str): test" in result
        assert "count (int): 5" in result

    def test_format_context_variables_empty(self):
        xerxes = Xerxes()
        assert xerxes.format_context_variables({}) == ""

    def test_format_context_variables_skips_callable(self):
        xerxes = Xerxes()
        result = xerxes.format_context_variables({"fn": lambda: 1})
        assert "fn" not in result

    def test_format_prompt(self):
        xerxes = Xerxes()
        assert xerxes.format_prompt("hello") == "hello"
        assert xerxes.format_prompt(None) == ""
        assert xerxes.format_prompt("") == ""

    def test_format_chat_history(self):
        xerxes = Xerxes()
        messages = MessagesHistory(messages=[UserMessage(content="hi"), AssistantMessage(content="ho")])
        result = xerxes.format_chat_history(messages)
        assert "user:" in result.lower() or "USER" in result
        assert "assistant:" in result.lower() or "ASSISTANT" in result

    def test_generate_function_section_empty(self):
        xerxes = Xerxes()
        assert xerxes.generate_function_section([]) == ""

    def test_generate_function_section_with_tool(self):
        xerxes = Xerxes()
        tool = _make_tool_fn("my_tool")
        section = xerxes.generate_function_section([tool])
        assert "my_tool" in section
        assert "Function:" in section

    def test_build_tool_prompt_label(self):
        xerxes = Xerxes()
        tool = _make_tool_fn("my_tool")
        label = xerxes._build_tool_prompt_label(tool)
        assert "my_tool" in label

    def test_build_tool_prompt_label_no_description(self):
        xerxes = Xerxes()

        def bare():
            pass

        bare.__name__ = "bare_tool"
        label = xerxes._build_tool_prompt_label(bare)
        assert label == "bare_tool"

    def test_add_depth(self):
        from xerxes.xerxes import add_depth

        assert add_depth("a\nb", add_prefix=True).startswith("  a")
        assert "\n  b" in add_depth("a\nb", add_prefix=True)


class TestXerxesStreamingNoFunctions:
    @pytest.mark.asyncio
    async def test_handle_streaming_sync_generator(self):
        llm = _FakeLLM(responses=[[_chunk(content="hello", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        chunks = []
        async for c in xerxes._handle_streaming(llm.responses[0], False, agent):
            chunks.append(c)
        assert any(c.content == "hello" for c in chunks if isinstance(c, StreamChunk))
        assert any(isinstance(c, Completion) for c in chunks)

    @pytest.mark.asyncio
    async def test_handle_streaming_async_generator(self):
        llm = _FakeLLM(responses=[[_chunk(content="hello", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")

        async def async_gen(resp, agent):
            for chunk in resp:
                yield chunk

        llm.astream_completion = async_gen
        chunks = []
        resp = await llm.generate_completion("")
        async for c in xerxes._handle_streaming(resp, False, agent):
            chunks.append(c)
        assert any(c.content == "hello" for c in chunks if isinstance(c, StreamChunk))

    @pytest.mark.asyncio
    async def test_handle_streaming_with_buffer(self):
        llm = _FakeLLM(responses=[[_chunk(content="hello", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        buf = StreamerBuffer()
        async for _ in xerxes._handle_streaming(llm.responses[0], False, agent, streamer_buffer=buf):
            pass
        buf.close()

    @pytest.mark.asyncio
    async def test_handle_streaming_records_error(self):
        class BadLLM(_FakeLLM):
            async def generate_completion(self, prompt, **kwargs):
                return []

            def stream_completion(self, response, agent=None):
                raise ValueError("stream error")

        llm = BadLLM(responses=[])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        with pytest.raises(ValueError, match="stream error"):
            async for _ in xerxes._handle_streaming([], False, agent):
                pass


class TestXerxesStreamingWithFunctions:
    @pytest.mark.asyncio
    async def test_handle_streaming_with_functions_no_calls(self):
        llm = _FakeLLM(responses=[[_chunk(content="plain", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        messages = MessagesHistory(messages=[])
        chunks = []
        async for c in xerxes._handle_streaming_with_functions(
            llm.responses[0], agent, {}, messages, True, False, False, None
        ):
            chunks.append(c)
        assert any(isinstance(c, Completion) for c in chunks)
        completion = next(c for c in chunks if isinstance(c, Completion))
        assert completion.final_content == "plain"

    @pytest.mark.asyncio
    async def test_handle_streaming_with_functions_tool_call(self):
        def echo(text: str) -> str:
            return text

        llm = _FakeLLM(
            responses=[
                [_chunk(function_calls=[{"id": "c1", "name": "echo", "arguments": {"text": "hi"}}], is_final=True)],
                [_chunk(content="done", is_final=True)],
            ]
        )
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake", functions=[echo])
        xerxes.register_agent(agent)
        messages = MessagesHistory(messages=[])
        chunks = []
        async for c in xerxes._handle_streaming_with_functions(
            llm.responses[0], agent, {}, messages, True, False, False, None
        ):
            chunks.append(c)
        assert any(isinstance(c, FunctionDetection) for c in chunks)
        assert any(isinstance(c, FunctionCallsExtracted) for c in chunks)
        assert any(isinstance(c, FunctionExecutionStart) for c in chunks)
        assert any(isinstance(c, FunctionExecutionComplete) for c in chunks)
        assert any(isinstance(c, ReinvokeSignal) for c in chunks)

    @pytest.mark.asyncio
    async def test_handle_streaming_with_functions_reinvoke_false(self):
        def echo(text: str) -> str:
            return text

        llm = _FakeLLM(
            responses=[
                [_chunk(function_calls=[{"id": "c1", "name": "echo", "arguments": {"text": "hi"}}], is_final=True)],
            ]
        )
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake", functions=[echo])
        xerxes.register_agent(agent)
        messages = MessagesHistory(messages=[])
        chunks = []
        async for c in xerxes._handle_streaming_with_functions(
            llm.responses[0], agent, {}, messages, False, False, False, None
        ):
            chunks.append(c)

        assert any(isinstance(c, Completion) for c in chunks)
        assert not any(isinstance(c, ReinvokeSignal) for c in chunks)

    @pytest.mark.asyncio
    async def test_handle_streaming_with_functions_agent_switch(self):
        def echo(text: str) -> str:
            return text

        llm = _FakeLLM(
            responses=[
                [_chunk(function_calls=[{"id": "c1", "name": "echo", "arguments": {"text": "hi"}}], is_final=True)],
                [_chunk(content="done", is_final=True)],
            ]
        )
        xerxes = Xerxes(llm=llm)
        a1 = Agent(id="a1", model="fake", functions=[echo])
        a2 = Agent(id="a2", model="fake", functions=[])
        xerxes.register_agent(a1)
        xerxes.register_agent(a2)

        xerxes.orchestrator.register_switch_trigger(type("Trigger", (), {}), lambda ctx, agents, current: "a2")
        messages = MessagesHistory(messages=[])
        chunks = []
        async for c in xerxes._handle_streaming_with_functions(
            llm.responses[0], a1, {}, messages, True, False, False, None
        ):
            chunks.append(c)
        assert any(isinstance(c, AgentSwitch) for c in chunks)

    @pytest.mark.asyncio
    async def test_handle_streaming_with_functions_streaming_tool_calls(self):
        llm = _FakeLLM(
            responses=[
                [
                    _chunk(content="", streaming_tool_calls={0: {"id": "t1", "name": "echo", "arguments": "{"}}),
                    _chunk(
                        content="",
                        streaming_tool_calls={0: {"id": "t1", "name": "echo", "arguments": '"text":"hi"}'}},
                        is_final=True,
                        function_calls=[{"id": "t1", "name": "echo", "arguments": {"text": "hi"}}],
                    ),
                ],
                [_chunk(content="done", is_final=True)],
            ]
        )
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake", functions=[lambda text: text])
        xerxes.register_agent(agent)
        messages = MessagesHistory(messages=[])
        chunks = []
        async for c in xerxes._handle_streaming_with_functions(
            llm.responses[0], agent, {}, messages, True, False, False, None
        ):
            chunks.append(c)
        stream_chunks = [c for c in chunks if isinstance(c, StreamChunk)]
        assert any(c.streaming_tool_calls for c in stream_chunks)

    @pytest.mark.asyncio
    async def test_handle_streaming_with_functions_tool_call_from_content(self):
        def echo(text: str) -> str:
            return text

        llm = _FakeLLM(
            responses=[
                [
                    _chunk(
                        content='<echo><arguments>{"text":"hi"}</arguments></echo>',
                        is_final=True,
                    )
                ],
                [_chunk(content="done", is_final=True)],
            ]
        )
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake", functions=[echo])
        xerxes.register_agent(agent)
        messages = MessagesHistory(messages=[])
        chunks = []
        async for c in xerxes._handle_streaming_with_functions(
            llm.responses[0], agent, {}, messages, True, False, False, None
        ):
            chunks.append(c)
        assert any(isinstance(c, FunctionDetection) for c in chunks)

    @pytest.mark.asyncio
    async def test_handle_streaming_with_functions_error_in_stream(self):
        class BadLLM(_FakeLLM):
            def stream_completion(self, response, agent=None):
                raise RuntimeError("boom")

        llm = BadLLM(responses=[])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        with pytest.raises(RuntimeError, match="boom"):
            async for _ in xerxes._handle_streaming_with_functions(
                [], agent, {}, MessagesHistory(messages=[]), True, False, False, None
            ):
                pass

    @pytest.mark.asyncio
    async def test_handle_streaming_with_functions_result_handoff(self):
        from xerxes.types.agent_types import Result

        def handoff_tool() -> Result:
            return Result(value="ok", agent=Agent(id="a2", model="fake"))

        llm = _FakeLLM(
            responses=[
                [_chunk(function_calls=[{"id": "c1", "name": "handoff_tool", "arguments": {}}], is_final=True)],
            ]
        )
        xerxes = Xerxes(llm=llm)
        a1 = Agent(id="a1", model="fake", functions=[handoff_tool])
        xerxes.register_agent(a1)
        messages = MessagesHistory(messages=[])
        async for _ in xerxes._handle_streaming_with_functions(
            llm.responses[0], a1, {}, messages, False, False, False, None
        ):
            pass
        assert xerxes.orchestrator.current_agent_id == "a2"


class TestXerxesCreateResponse:
    @pytest.mark.asyncio
    async def test_create_response_streaming(self):
        llm = _FakeLLM(responses=[[_chunk(content="hi", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        result = await xerxes.create_response(prompt="hello", agent_id=agent, stream=True)
        chunks = []
        async for c in result:
            chunks.append(c)
        assert any(isinstance(c, Completion) for c in chunks)

    @pytest.mark.asyncio
    async def test_create_response_non_streaming(self):
        llm = _FakeLLM(responses=[[_chunk(content="hi", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        result = await xerxes.create_response(prompt="hello", agent_id=agent, stream=False)
        assert isinstance(result, ResponseResult)
        assert result.content == "hi"

    @pytest.mark.asyncio
    async def test_create_response_with_agent_id_string(self):
        llm = _FakeLLM(responses=[[_chunk(content="hi", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        result = await xerxes.create_response(prompt="hello", agent_id="a1", stream=False)
        assert result.content == "hi"

    @pytest.mark.asyncio
    async def test_create_response_apply_functions_false(self):
        llm = _FakeLLM(responses=[[_chunk(content="hi", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        result = await xerxes.create_response(prompt="hello", agent_id=agent, stream=False, apply_functions=False)
        assert result.content == "hi"

    @pytest.mark.asyncio
    async def test_create_response_use_instructed_prompt(self):
        llm = _FakeLLM(responses=[[_chunk(content="hi", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        result = await xerxes.create_response(prompt="hello", agent_id=agent, stream=False, use_instructed_prompt=True)
        assert result.content == "hi"

        assert isinstance(llm.calls[0]["prompt"], str)

    @pytest.mark.asyncio
    async def test_create_response_with_messages(self):
        llm = _FakeLLM(responses=[[_chunk(content="hi", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        history = MessagesHistory(messages=[UserMessage(content="prior")])
        result = await xerxes.create_response(prompt="hello", agent_id=agent, messages=history, stream=False)
        assert result.content == "hi"

    @pytest.mark.asyncio
    async def test_create_response_setup_error(self):
        class BadLLM(_FakeLLM):
            async def generate_completion(self, prompt, **kwargs):
                raise ConnectionError("fail")

        llm = BadLLM(responses=[])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        with pytest.raises(ConnectionError, match="fail"):
            await xerxes.create_response(prompt="hello", agent_id=agent, stream=False)

    @pytest.mark.asyncio
    async def test_create_response_with_streamer_buffer(self):
        llm = _FakeLLM(responses=[[_chunk(content="hi", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        buf = StreamerBuffer()
        result = await xerxes.create_response(prompt="hello", agent_id=agent, stream=False, streamer_buffer=buf)
        assert result.content == "hi"

    @pytest.mark.asyncio
    async def test_create_response_reinvoked_runtime(self):
        llm = _FakeLLM(responses=[[_chunk(content="hi", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        result = await xerxes.create_response(prompt="hello", agent_id=agent, stream=False, reinvoked_runtime=True)
        assert result.content == "hi"


class TestXerxesRun:
    def test_run_streaming(self):
        llm = _FakeLLM(responses=[[_chunk(content="hi", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        gen = xerxes.run(prompt="hello", agent_id=agent, stream=True)
        chunks = list(gen)
        assert any(isinstance(c, Completion) for c in chunks)

    def test_run_non_streaming(self):
        llm = _FakeLLM(responses=[[_chunk(content="hi", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        result = xerxes.run(prompt="hello", agent_id=agent, stream=False)
        assert isinstance(result, ResponseResult)
        assert result.content == "hi"

    def test_run_non_streaming_with_tool_execution(self):
        def echo(text: str) -> str:
            return text

        llm = _FakeLLM(
            responses=[
                [_chunk(function_calls=[{"id": "c1", "name": "echo", "arguments": {"text": "hi"}}], is_final=True)],
                [_chunk(content="done", is_final=True)],
            ]
        )
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake", functions=[echo])
        xerxes.register_agent(agent)
        result = xerxes.run(prompt="hello", agent_id=agent, stream=False)
        assert isinstance(result, ResponseResult)
        assert result.content == "done"

    def test_run_with_context_variables(self):
        llm = _FakeLLM(responses=[[_chunk(content="hi", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        result = xerxes.run(prompt="hello", agent_id=agent, stream=False, context_variables={"key": "val"})
        assert result.content == "hi"

    def test_run_with_print_formatted_prompt(self, capsys):
        llm = _FakeLLM(responses=[[_chunk(content="hi", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        result = xerxes.run(prompt="hello", agent_id=agent, stream=False, print_formatted_prompt=True)
        assert result.content == "hi"
        captured = capsys.readouterr()
        assert "user" in captured.out.lower() or captured.out == ""

    def test_run_streaming_exception_propagated(self):
        class BadLLM(_FakeLLM):
            async def generate_completion(self, prompt, **kwargs):
                raise ValueError("bad")

        llm = BadLLM(responses=[])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        gen = xerxes.run(prompt="hello", agent_id=agent, stream=True)
        with pytest.raises(ValueError, match="bad"):
            list(gen)


class TestXerxesThreadRun:
    def test_thread_run_returns_buffer_and_thread(self):
        llm = _FakeLLM(responses=[[_chunk(content="hi", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        buf, thread = xerxes.thread_run(prompt="hello", agent_id=agent)
        assert isinstance(buf, StreamerBuffer)
        assert isinstance(thread, threading.Thread)
        thread.join(timeout=5)
        result = buf.get_result(timeout=5)
        assert isinstance(result, ResponseResult)
        assert result.content == "hi"

    def test_thread_run_with_pre_created_buffer(self):
        llm = _FakeLLM(responses=[[_chunk(content="hi", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        buf = StreamerBuffer()
        buf2, thread = xerxes.thread_run(prompt="hello", agent_id=agent, streamer_buffer=buf)
        assert buf2 is buf
        thread.join(timeout=5)

    def test_thread_run_exception_captured(self):
        class BadLLM(_FakeLLM):
            async def generate_completion(self, prompt, **kwargs):
                raise RuntimeError("boom")

        llm = BadLLM(responses=[])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        buf, thread = xerxes.thread_run(prompt="hello", agent_id=agent)
        thread.join(timeout=5)
        with pytest.raises(RuntimeError, match="boom"):
            buf.get_result(timeout=5)

    @pytest.mark.asyncio
    async def test_athread_run_returns_buffer_and_task(self):
        llm = _FakeLLM(responses=[[_chunk(content="hi", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        buf, task = await xerxes.athread_run(prompt="hello", agent_id=agent)
        assert isinstance(buf, StreamerBuffer)
        assert isinstance(task, asyncio.Task)
        result = await buf.aget_result()
        assert isinstance(result, ResponseResult)
        assert result.content == "hi"

    @pytest.mark.asyncio
    async def test_athread_run_exception_captured(self):
        class BadLLM(_FakeLLM):
            async def generate_completion(self, prompt, **kwargs):
                raise RuntimeError("boom")

        llm = BadLLM(responses=[])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        buf, _task = await xerxes.athread_run(prompt="hello", agent_id=agent)
        with pytest.raises(RuntimeError, match="boom"):
            await buf.aget_result()


class TestXerxesReinvoke:
    def test_build_reinvoke_messages(self):
        xerxes = Xerxes()
        original = MessagesHistory(messages=[UserMessage(content="hello")])
        fc = RequestFunctionCall(name="echo", arguments={"text": "hi"}, id="c1")
        result = RequestFunctionCall(name="echo", arguments={"text": "hi"}, id="c1")
        result.status = ExecutionStatus.SUCCESS
        result.result = "hi"
        messages = xerxes._build_reinvoke_messages(original, "assistant says", [fc], [result])
        roles = [m.role for m in messages.messages]
        assert "assistant" in roles
        assert "tool" in roles
        assert "user" in roles
        assert messages.messages[-1].content == xerxes.REINVOKE_FOLLOWUP_INSTRUCTION

    def test_build_reinvoke_messages_with_tool_result_hook(self):
        xerxes = Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=True))
        original = MessagesHistory(messages=[UserMessage(content="hello")])
        fc = RequestFunctionCall(name="echo", arguments={"text": "hi"}, id="c1")
        result = RequestFunctionCall(name="echo", arguments={"text": "hi"}, id="c1")
        result.status = ExecutionStatus.SUCCESS
        result.result = "hi"
        messages = xerxes._build_reinvoke_messages(original, "assistant says", [fc], [result])

        assert any(isinstance(m, ToolMessage) for m in messages.messages)

    def test_compact_reinvoke_history(self):
        xerxes = Xerxes()
        msgs = [
            UserMessage(content="hello"),
            AssistantMessage(
                content="hi", tool_calls=[ToolCall(id="t1", function=FunctionCall(name="f", arguments="{}"))]
            ),
            ToolMessage(content="result", tool_call_id="t1"),
            UserMessage(content=xerxes.REINVOKE_FOLLOWUP_INSTRUCTION),
        ]
        compacted = xerxes._compact_reinvoke_history(msgs)
        assert len(compacted) == 1
        assert compacted[0].content == "hello"

    def test_compact_reinvoke_history_with_operator_attachment(self):
        xerxes = Xerxes()

        chunk = TextChunk(text="[TOOL IMAGE RESULT] something")
        msgs = [
            UserMessage(content="hello"),
            AssistantMessage(content="hi"),
            UserMessage(content=[chunk]),
            UserMessage(content=xerxes.REINVOKE_FOLLOWUP_INSTRUCTION),
        ]
        compacted = xerxes._compact_reinvoke_history(msgs)
        assert len(compacted) == 2
        assert compacted[0].content == "hello"
        assert compacted[1].content == "hi"

    def test_is_reinvoke_followup_message(self):
        msg = UserMessage(content=Xerxes.REINVOKE_FOLLOWUP_INSTRUCTION)
        assert Xerxes._is_reinvoke_followup_message(msg) is True

    def test_is_reinvoke_followup_message_not_matching(self):
        msg = UserMessage(content="something else")
        assert Xerxes._is_reinvoke_followup_message(msg) is False

    def test_is_operator_reinvoke_attachment_true(self):
        chunk = TextChunk(text="[TOOL IMAGE RESULT] img")
        msg = UserMessage(content=[chunk])
        assert Xerxes._is_operator_reinvoke_attachment(msg) is True

    def test_is_operator_reinvoke_attachment_false(self):
        msg = UserMessage(content="plain text")
        assert Xerxes._is_operator_reinvoke_attachment(msg) is False


class TestXerxesMemory:
    def test_update_memory_from_prompt_when_enabled(self):
        xerxes = Xerxes(enable_memory=True)
        xerxes.memory_store.add_memory = MagicMock()
        xerxes._update_memory_from_prompt("hello", "a1")
        xerxes.memory_store.add_memory.assert_called_once()
        call_kwargs = xerxes.memory_store.add_memory.call_args.kwargs
        assert "hello" in call_kwargs["content"]
        assert call_kwargs["memory_type"].value == "short_term"

    def test_update_memory_from_prompt_when_disabled(self):
        xerxes = Xerxes(enable_memory=False)

        xerxes._update_memory_from_prompt("hello", "a1")

    def test_update_memory_from_response_when_enabled(self):
        xerxes = Xerxes(enable_memory=True)
        xerxes.memory_store.add_memory = MagicMock()
        fc = RequestFunctionCall(name="echo", arguments={"text": "hi"}, id="c1")
        xerxes._update_memory_from_response("hello", "a1", function_calls=[fc])
        assert xerxes.memory_store.add_memory.call_count == 2

    def test_update_memory_from_response_when_disabled(self):
        xerxes = Xerxes(enable_memory=False)
        xerxes._update_memory_from_response("hello", "a1")

    def test_add_memory_tools_to_agent(self):
        xerxes = Xerxes(enable_memory=True)
        agent = Agent(id="a1", model="fake", functions=[])
        xerxes._add_memory_tools_to_agent(agent)
        assert len(agent.functions) > 0

    def test_add_memory_tools_skips_duplicates(self):
        xerxes = Xerxes(enable_memory=True)
        from xerxes.tools.memory_tool import MEMORY_TOOLS

        agent = Agent(id="a1", model="fake", functions=list(MEMORY_TOOLS))
        original_count = len(agent.functions)
        xerxes._add_memory_tools_to_agent(agent)
        assert len(agent.functions) == original_count


class TestXerxesRuntimeTurnState:
    def test_new_runtime_turn_id(self):
        tid = Xerxes._new_runtime_turn_id()
        assert isinstance(tid, str)
        assert len(tid) == 12

    def test_append_turn_tool_results_none(self):
        xerxes = Xerxes()
        xerxes._append_turn_tool_results(None, [RequestFunctionCall(name="f", arguments={})])

    def test_finalize_runtime_turn_no_runtime_state(self):
        xerxes = Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=False))
        xerxes._finalize_runtime_turn("a1", "content")

    def test_record_runtime_error_no_runtime_state(self):
        xerxes = Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=False))
        xerxes._record_runtime_error("a1", ValueError("oops"), "ctx")

    def test_notify_turn_start_no_hooks(self):
        xerxes = Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=False))
        xerxes._notify_turn_start("a1")

    def test_notify_turn_end_no_hooks(self):
        xerxes = Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=False))
        xerxes._notify_turn_end("a1")

    def test_notify_runtime_error_no_hooks(self):
        xerxes = Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=False))
        xerxes._notify_runtime_error("a1", ValueError("oops"))


class TestXerxesSubagentManager:
    def test_create_subagent_manager_returns_manager(self):
        xerxes = Xerxes()
        mgr = xerxes.create_subagent_manager()
        assert mgr is not None
        assert hasattr(mgr, "set_runner")

    def test_create_subagent_manager_with_whitelist(self):
        xerxes = Xerxes()
        mgr = xerxes.create_subagent_manager()

        task = mgr.spawn(
            prompt="test",
            config={"_tools_whitelist": ["nonexistent"]},
            system_prompt="sys",
        )

        mgr.wait(task.id)


class TestXerxesBridgeMethods:
    def test_create_query_engine(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="gpt-4", instructions="help")
        xerxes.register_agent(agent)
        engine = xerxes.create_query_engine()
        assert engine is not None

    def test_create_query_engine_with_overrides(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="gpt-4", instructions="help")
        xerxes.register_agent(agent)
        engine = xerxes.create_query_engine(model="gpt-3.5", system_prompt="override")
        assert engine is not None

    def test_create_runtime_session(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="gpt-4")
        xerxes.register_agent(agent)
        session = xerxes.create_runtime_session(prompt="test")
        assert session is not None
        assert session.prompt == "test"

    def test_create_runtime_session_no_agent(self):
        xerxes = Xerxes()
        session = xerxes.create_runtime_session(prompt="test")
        assert session.prompt == "test"

    def test_get_execution_registry(self):
        xerxes = Xerxes()
        registry = xerxes.get_execution_registry()
        assert registry is not None

    def test_get_tool_executor(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        executor = xerxes.get_tool_executor()
        assert callable(executor)

    def test_bootstrap(self):
        xerxes = Xerxes()
        result = xerxes.bootstrap(extra_context="extra")
        assert result is not None


class TestXerxesEdgeCases:
    def test_manage_messages_agent_with_functions_not_instructed(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="fake", instructions="help", functions=[_make_tool_fn()])
        messages = xerxes.manage_messages(agent=agent, prompt="hello")
        system = messages.messages[0].content
        assert "Do not use functions for greetings" in system

    def test_manage_messages_agent_with_functions_instructed(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="fake", instructions="help", functions=[_make_tool_fn()])
        messages = xerxes.manage_messages(agent=agent, prompt="hello", use_instructed_prompt=True)
        system = messages.messages[0].content
        assert "Do not call a function for greetings" in system

    def test_manage_messages_with_list_rules(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="fake", instructions="help", rules=["rule1", "rule2"])
        messages = xerxes.manage_messages(agent=agent, prompt="hello")
        system = messages.messages[0].content
        assert "rule1" in system
        assert "rule2" in system

    def test_manage_messages_with_callable_rules(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="fake", instructions="help", rules=lambda: ["single_rule"])
        messages = xerxes.manage_messages(agent=agent, prompt="hello")
        system = messages.messages[0].content
        assert "single_rule" in system

    def test_merge_system_history_no_messages(self):
        xerxes = Xerxes()
        merged, remaining = xerxes._merge_system_history("base", None)
        assert merged == "base"
        assert remaining == []

    def test_merge_system_history_empty_messages(self):
        xerxes = Xerxes()
        merged, remaining = xerxes._merge_system_history("base", MessagesHistory(messages=[]))
        assert merged == "base"
        assert remaining == []

    def test_extract_from_markdown_bad_json(self):
        result = Xerxes.extract_from_markdown("json", "```json\nnot json\n```")
        assert result == "not json"

    def test_extract_function_calls_tool_calls_exception(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="fake", functions=[])
        tc = ToolCall(id="t1", function=FunctionCall(name="missing", arguments="{}"))
        calls = xerxes._extract_function_calls("", agent, tool_calls=[tc])
        assert len(calls) == 0

    def test_create_response_no_agent_raises(self):
        xerxes = Xerxes()

        with pytest.raises(Exception):  # noqa: B017
            asyncio.get_event_loop().run_until_complete(xerxes.create_response(prompt="hello", stream=False))

    def test_run_stream_empty_response(self):
        llm = _FakeLLM(responses=[[_chunk(content="", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        result = xerxes.run(prompt="hello", agent_id=agent, stream=False)
        assert result.content == ""

    def test_handle_streaming_with_functions_no_agent_functions(self):
        llm = _FakeLLM(
            responses=[
                [_chunk(content="<unknown><arguments>{}</arguments></unknown>", is_final=True)],
            ]
        )
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake", functions=[])
        xerxes.register_agent(agent)
        chunks = []

        async def collect():
            async for c in xerxes._handle_streaming_with_functions(
                llm.responses[0], agent, {}, MessagesHistory(messages=[]), True, False, False, None
            ):
                chunks.append(c)

        asyncio.get_event_loop().run_until_complete(collect())

        assert any(isinstance(c, Completion) for c in chunks)


class TestXerxesAdditionalCoverage:
    def test_system_message_to_text_with_list_content(self):
        msg = SystemMessage(content=[TextChunk(text="  part1  "), TextChunk(text="  "), TextChunk(text="part2")])
        result = Xerxes._system_message_to_text(msg)
        assert "part1" in result
        assert "part2" in result

    def test_system_message_to_text_all_empty_parts(self):
        msg = SystemMessage(content=[TextChunk(text="  ")])
        assert Xerxes._system_message_to_text(msg) is None

    def test_generate_function_section_categorized(self):
        xerxes = Xerxes()
        tool = _make_tool_fn("cat_tool")
        tool.category = "Math"  # type: ignore
        section = xerxes.generate_function_section([tool])
        assert "## Math Functions" in section
        assert "cat_tool" in section

    def test_generate_function_section_with_uncategorized_after_category(self):
        xerxes = Xerxes()
        cat_tool = _make_tool_fn("cat_tool")
        cat_tool.category = "Math"  # type: ignore
        plain_tool = _make_tool_fn("plain_tool")
        section = xerxes.generate_function_section([cat_tool, plain_tool])
        assert "## Math Functions" in section
        assert "## Other Functions" in section

    def test_generate_function_section_parsing_error(self):
        xerxes = Xerxes()

        def bad_tool():
            pass

        bad_tool.__name__ = "bad_tool"
        with patch("xerxes.xerxes.function_to_json", side_effect=ValueError("bad")):
            section = xerxes.generate_function_section([bad_tool])
        assert "Warning: Unable to parse function bad_tool" in section

    def test_format_function_doc_with_returns(self):
        xerxes = Xerxes()
        schema = {
            "name": "test_fn",
            "description": "A test fn.",
            "parameters": {"properties": {}, "required": []},
            "returns": "str",
        }
        doc = xerxes._format_function_doc(schema)
        assert "Returns" in doc

    def test_format_function_doc_with_examples(self):
        xerxes = Xerxes()
        schema = {
            "name": "test_fn",
            "description": "A test fn.",
            "parameters": {"properties": {}, "required": []},
            "examples": [{"query": "hello"}],
        }
        doc = xerxes._format_function_doc(schema)
        assert "Examples:" in doc
        assert "query" in doc

    def test_format_function_doc_with_enum(self):
        xerxes = Xerxes()
        schema = {
            "name": "test_fn",
            "description": "A test fn.",
            "parameters": {
                "properties": {
                    "mode": {"type": "string", "description": "Mode", "enum": ["a", "b"]},
                },
                "required": ["mode"],
            },
        }
        doc = xerxes._format_function_doc(schema)
        assert "Allowed values" in doc

    def test_build_tool_prompt_label_long_description(self):
        xerxes = Xerxes()

        def long_desc():
            pass

        long_desc.__name__ = "long_tool"
        long_desc.__doc__ = "A" + " very long description" * 20
        label = xerxes._build_tool_prompt_label(long_desc)
        assert label.endswith("...")

    def test_build_tool_prompt_label_exception(self):
        xerxes = Xerxes()

        def bad_tool():
            pass

        bad_tool.__name__ = "bad_tool"
        with patch("xerxes.xerxes.function_to_json", side_effect=ValueError("bad")):
            label = xerxes._build_tool_prompt_label(bad_tool)
        assert label == "bad_tool"

    def test_detect_function_calls_regex_no_match(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="fake", functions=[_make_tool_fn("my_tool")])
        assert xerxes._detect_function_calls_regex("no tools here", agent) is False

    def test_create_response_print_formatted_prompt(self, capsys):
        llm = _FakeLLM(responses=[[_chunk(content="hi", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        result = asyncio.get_event_loop().run_until_complete(
            xerxes.create_response(prompt="hello", agent_id=agent, stream=False, print_formatted_prompt=True)
        )
        assert result.content == "hi"

    def test_create_response_print_formatted_prompt_instructed(self, capsys):
        llm = _FakeLLM(responses=[[_chunk(content="hi", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        result = asyncio.get_event_loop().run_until_complete(
            xerxes.create_response(
                prompt="hello", agent_id=agent, stream=False, print_formatted_prompt=True, use_instructed_prompt=True
            )
        )
        assert result.content == "hi"

    @pytest.mark.asyncio
    async def test_handle_streaming_with_functions_aiter_path(self):
        """Cover the __aiter__ branch in _handle_streaming_with_functions."""
        llm = _FakeLLM(responses=[[_chunk(content="async", is_final=True)]])

        class AiterResponse:
            def __init__(self, items):
                self._items = items

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._items:
                    raise StopAsyncIteration
                return self._items.pop(0)

        async def fake_gen(resp, agent):
            async for item in resp:
                yield item

        llm.astream_completion = fake_gen
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        resp = AiterResponse([_chunk(content="async", is_final=True)])
        chunks = []
        async for c in xerxes._handle_streaming_with_functions(
            resp, agent, {}, MessagesHistory(messages=[]), True, False, False, None
        ):
            chunks.append(c)
        assert any(isinstance(c, Completion) for c in chunks)

    @pytest.mark.asyncio
    async def test_handle_streaming_aiter_path(self):
        """Cover the __aiter__ branch in _handle_streaming."""

        class AiterResponse:
            def __init__(self, items):
                self._items = items

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._items:
                    raise StopAsyncIteration
                return self._items.pop(0)

        llm = _FakeLLM(responses=[])

        async def fake_gen(resp, agent):
            async for item in resp:
                yield item

        llm.astream_completion = fake_gen
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        resp = AiterResponse([_chunk(content="async", is_final=True)])
        chunks = []
        async for c in xerxes._handle_streaming(resp, False, agent):
            chunks.append(c)
        assert any(isinstance(c, Completion) for c in chunks)

    def test_run_stream_queue_timeout(self):
        """Cover queue-empty branch in _run_stream."""
        llm = _FakeLLM(responses=[[_chunk(content="x", is_final=True)]])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        gen = xerxes._run_stream(prompt="hi", agent_id=agent)
        chunks = list(gen)
        assert any(isinstance(c, StreamChunk) and c.content == "x" for c in chunks)

    def test_add_memory_tools_skips_when_already_present(self):
        xerxes = Xerxes(enable_memory=True)
        from xerxes.tools.memory_tool import MEMORY_TOOLS

        agent = Agent(id="a1", model="fake", functions=list(MEMORY_TOOLS))
        xerxes._add_memory_tools_to_agent(agent)

        assert len(agent.functions) == len(MEMORY_TOOLS)

    def test_manage_messages_with_empty_rules_list(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="fake", instructions="help", rules=[])
        messages = xerxes.manage_messages(agent=agent, prompt="hello")

        assert len(messages.messages) >= 1

    def test_manage_messages_agent_functions_none(self):
        xerxes = Xerxes()
        agent = Agent(id="a1", model="fake", instructions="help", functions=[])
        messages = xerxes.manage_messages(agent=agent, prompt="hello")
        assert len(messages.messages) >= 1

    def test_format_function_parameters_with_enum(self):
        xerxes = Xerxes()
        params = {
            "properties": {
                "mode": {"type": "string", "enum": ["a", "b"]},
            },
            "required": ["mode"],
        }
        result = xerxes.format_function_parameters(params)
        assert "Allowed values" in result

    def test_format_function_parameters_skip_context_variables(self):
        xerxes = Xerxes()
        params = {
            "properties": {
                "context_variables": {"type": "object"},
                "query": {"type": "string"},
            },
            "required": ["query"],
        }
        result = xerxes.format_function_parameters(params)
        assert "context_variables" not in result
        assert "query" in result

    def test_build_reinvoke_messages_failed_result(self):
        xerxes = Xerxes()
        original = MessagesHistory(messages=[UserMessage(content="hello")])
        fc = RequestFunctionCall(name="echo", arguments={"text": "hi"}, id="c1")
        result = RequestFunctionCall(name="echo", arguments={"text": "hi"}, id="c1")
        result.status = ExecutionStatus.FAILURE
        result.error = "boom"
        messages = xerxes._build_reinvoke_messages(original, "assistant says", [fc], [result])
        tool_msgs = [m for m in messages.messages if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert "Error: boom" in tool_msgs[0].content

    def test_build_reinvoke_messages_no_tool_calls(self):
        xerxes = Xerxes()
        original = MessagesHistory(messages=[UserMessage(content="hello")])
        messages = xerxes._build_reinvoke_messages(original, "assistant says", [], [])
        assert isinstance(messages.messages[-1], UserMessage)

    @pytest.mark.asyncio
    async def test_handle_streaming_with_functions_streaming_tool_calls_dict(self):
        """Cover streaming_tool_calls as list path."""
        llm = _FakeLLM(
            responses=[
                [
                    _chunk(content="", streaming_tool_calls=[{"id": "t1", "name": "echo", "arguments": "{"}]),
                    _chunk(
                        content="",
                        streaming_tool_calls=[{"id": "t1", "name": "echo", "arguments": '"text":"hi"}'}],
                        is_final=True,
                        function_calls=[{"id": "t1", "name": "echo", "arguments": {"text": "hi"}}],
                    ),
                ],
                [_chunk(content="done", is_final=True)],
            ]
        )
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake", functions=[lambda text: text])
        xerxes.register_agent(agent)
        messages = MessagesHistory(messages=[])
        chunks = []
        async for c in xerxes._handle_streaming_with_functions(
            llm.responses[0], agent, {}, messages, True, False, False, None
        ):
            chunks.append(c)
        stream_chunks = [c for c in chunks if isinstance(c, StreamChunk)]
        assert any(c.streaming_tool_calls for c in stream_chunks)

    def test_run_stream_with_exception_holder(self):
        class BadLLM(_FakeLLM):
            async def generate_completion(self, prompt, **kwargs):
                raise ValueError("boom")

        llm = BadLLM(responses=[])
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        gen = xerxes._run_stream(prompt="hi", agent_id=agent)
        with pytest.raises(ValueError, match="boom"):
            list(gen)

    def test_compact_reinvoke_history_empty(self):
        xerxes = Xerxes()
        compacted = xerxes._compact_reinvoke_history([])
        assert compacted == []

    def test_compact_reinvoke_history_no_trailing(self):
        xerxes = Xerxes()
        msgs = [UserMessage(content="hello"), AssistantMessage(content="hi")]
        compacted = xerxes._compact_reinvoke_history(msgs)
        assert len(compacted) == 2

    def test_extract_function_calls_from_tagged_markup_no_body(self):
        xerxes = Xerxes()
        tool = _make_tool_fn("my_tool")
        agent = Agent(id="a1", model="fake", functions=[tool])
        content = "<function=my_tool>  </function>"
        calls = xerxes._extract_function_calls_from_tagged_markup(content, agent)
        assert len(calls) == 0

    def test_extract_function_calls_from_tagged_markup_no_required_fields(self):
        xerxes = Xerxes()

        def free_tool():
            pass

        free_tool.__name__ = "free_tool"
        free_tool.__doc__ = "No args."
        agent = Agent(id="a1", model="fake", functions=[free_tool])
        content = "<function=free_tool></function>"
        calls = xerxes._extract_function_calls_from_tagged_markup(content, agent)
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_create_response_non_streaming_no_functions_collects_reasoning(self):
        llm = _FakeLLM(
            responses=[
                [
                    {
                        "content": "r1",
                        "buffered_content": "r1",
                        "reasoning_content": "think",
                        "buffered_reasoning_content": "think",
                        "is_final": False,
                        "function_calls": [],
                        "tool_calls": None,
                        "streaming_tool_calls": None,
                        "raw_chunk": None,
                    },
                    {
                        "content": "r2",
                        "buffered_content": "r1r2",
                        "reasoning_content": None,
                        "buffered_reasoning_content": "think",
                        "is_final": True,
                        "function_calls": [],
                        "tool_calls": None,
                        "streaming_tool_calls": None,
                        "raw_chunk": None,
                    },
                ]
            ]
        )
        xerxes = Xerxes(llm=llm)
        agent = Agent(id="a1", model="fake")
        xerxes.register_agent(agent)
        result = await xerxes.create_response(prompt="hello", agent_id=agent, stream=False, apply_functions=False)
        assert result.content == "r1r2"
        assert result.reasoning_content == "think"
