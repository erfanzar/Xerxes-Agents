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
"""Runtime integration tests for the opt-in OpenClaw-style feature layer."""

from __future__ import annotations

from pathlib import Path

import pytest
from xerxes import Agent, AgentRuntimeOverrides, RuntimeFeaturesConfig, Xerxes, create_llm
from xerxes.extensions.plugins import PluginRegistry
from xerxes.llms.base import BaseLLM, LLMConfig
from xerxes.runtime.loop_detection import LoopDetectionConfig
from xerxes.security.policy import ToolPolicy
from xerxes.security.sandbox import SandboxConfig, SandboxMode
from xerxes.tools.standalone import WriteFile
from xerxes.types import ExecutionStatus, RequestFunctionCall


def _chunk(*, content: str | None = None, function_calls: list[dict] | None = None, is_final: bool = False) -> dict:
    buffered_content = content or ""
    return {
        "content": content,
        "buffered_content": buffered_content,
        "reasoning_content": None,
        "buffered_reasoning_content": "",
        "function_calls": function_calls or [],
        "tool_calls": None,
        "streaming_tool_calls": None,
        "raw_chunk": None,
        "is_final": is_final,
    }


class _FakeLLM(BaseLLM):
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


def _write_runtime_plugin(path: Path, *, with_hooks: bool = False) -> None:
    hook_block = ""
    register_block = '    registry.register_tool("plugin_echo", plugin_echo, meta=PLUGIN_META)\n'
    if with_hooks:
        hook_block = """
def before_tool_call(tool_name, arguments, agent_id):
    if tool_name != "plugin_echo":
        return arguments
    updated = arguments.copy()
    updated["text"] = f"HOOKED-{updated['text']}"
    return updated


def after_tool_call(tool_name, arguments, result, agent_id):
    if tool_name != "plugin_echo":
        return result
    return f"{result}:after"


def tool_result_persist(tool_name, result, agent_id):
    if tool_name != "plugin_echo":
        return result
    return str(result).upper()

"""
        register_block += (
            '    registry.register_hook("before_tool_call", before_tool_call, plugin_name=PLUGIN_META.name)\n'
        )
        register_block += (
            '    registry.register_hook("after_tool_call", after_tool_call, plugin_name=PLUGIN_META.name)\n'
        )
        register_block += (
            '    registry.register_hook("tool_result_persist", tool_result_persist, plugin_name=PLUGIN_META.name)\n'
        )

    path.write_text(
        f"""from xerxes.extensions.plugins import PluginMeta, PluginType

PLUGIN_META = PluginMeta(name="runtime_plugin", version="1.0.0", plugin_type=PluginType.TOOL)

def plugin_echo(text: str) -> str:
    return f"echo:{{text}}"

def bootstrap_files(agent_id):
    return "Bootstrap hook from plugin"

{hook_block}
def register(registry):
{register_block}    registry.register_hook("bootstrap_files", bootstrap_files, plugin_name=PLUGIN_META.name)
"""
    )


def _write_skill(path: Path) -> None:
    path.write_text(
        """---
name: research
description: Research and synthesize findings
version: "1.0"
tags: [research]
---

# Research Skill

Break the task into smaller questions and synthesize the answers.
"""
    )


def test_runtime_prompt_lists_tool_summaries_instead_of_names_only():
    xerxes = Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=True))
    agent = Agent(id="writer", model="fake", instructions="Help with files.", functions=[WriteFile])

    messages = xerxes.manage_messages(agent=agent, prompt="write a report")
    system_prompt = messages.to_openai()["messages"][0]["content"]

    assert "WriteFile:" in system_prompt
    assert "Write text to a file" in system_prompt


def test_xerxes_enables_operator_runtime_by_default():
    xerxes = Xerxes()

    assert xerxes._runtime_features_state is not None
    assert xerxes._runtime_features_state.operator_state is not None
    assert xerxes._runtime_features_state.operator_state.config.power_tools_enabled is True
    assert xerxes.runtime_features.enabled is True


def _tool_messages(messages: list[dict]) -> list[dict]:
    return [message for message in messages if message["role"] == "tool"]


def test_runtime_disabled_keeps_existing_behavior(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_runtime_plugin(plugin_dir / "runtime_plugin.py")

    skill_dir = tmp_path / "skills" / "research"
    skill_dir.mkdir(parents=True)
    _write_skill(skill_dir / "SKILL.md")

    xerxes = Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=False))
    agent = Agent(id="plain", model="fake", instructions="Base instructions", functions=[])
    xerxes.register_agent(agent)

    assert "plugin_echo" not in agent.get_available_functions()
    system_content = xerxes.manage_messages(agent=agent, prompt="hello").messages[0].content
    assert "[Runtime Context]" not in system_content
    assert "Bootstrap hook from plugin" not in system_content
    assert "Research Skill" not in system_content


def test_runtime_enabled_discovers_extensions_and_enriches_prompt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_runtime_plugin(plugin_dir / "runtime_plugin.py")

    skill_dir = tmp_path / "skills" / "research"
    skill_dir.mkdir(parents=True)
    _write_skill(skill_dir / "SKILL.md")

    xerxes = Xerxes(
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            enabled_skills=["research"],
            guardrails=["Be safe"],
        )
    )
    agent = Agent(id="runtime", model="fake", instructions="Base instructions", functions=[])
    xerxes.register_agent(agent)

    assert "plugin_echo" in agent.get_available_functions()
    system_content = xerxes.manage_messages(agent=agent, prompt="hello").messages[0].content
    assert system_content.startswith("[Identity]\n")
    assert "[Identity]" in system_content
    assert "[Tooling]" in system_content
    assert "[Safety]" in system_content
    assert "[Runtime Context]" in system_content
    assert "[Workspace]" in system_content
    assert "[Skills]" in system_content
    assert "[Enabled Skill Instructions]" in system_content
    assert "Research Skill" in system_content
    assert "Bootstrap hook from plugin" in system_content
    assert "Be safe" in system_content
    assert "plugin_echo" in system_content


def test_runtime_workspace_is_pinned_to_launch_directory(tmp_path, monkeypatch):
    launch_root = tmp_path / "launch"
    other_root = tmp_path / "other"
    launch_root.mkdir()
    other_root.mkdir()

    monkeypatch.chdir(launch_root)
    xerxes = Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=True))
    agent = Agent(id="runtime", model="fake", instructions="Base instructions", functions=[])
    xerxes.register_agent(agent)

    monkeypatch.chdir(other_root)
    system_content = xerxes.manage_messages(agent=agent, prompt="where am I").messages[0].content

    assert f"Directory: {launch_root}" in system_content
    assert f"Directory: {other_root}" not in system_content
    assert xerxes._runtime_features_state is not None
    assert xerxes._runtime_features_state.operator_state is not None
    assert xerxes._runtime_features_state.operator_state.config.shell_default_workdir == str(launch_root)


@pytest.mark.asyncio
async def test_runtime_hooks_and_plugin_tool_apply_in_live_execution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    _write_runtime_plugin(plugin_dir / "runtime_plugin.py", with_hooks=True)

    llm = _FakeLLM(
        responses=[
            [
                _chunk(
                    function_calls=[{"id": "call_1", "name": "plugin_echo", "arguments": {"text": "hi"}}], is_final=True
                )
            ],
            [_chunk(content="final", is_final=True)],
        ]
    )
    xerxes = Xerxes(llm=llm, runtime_features=RuntimeFeaturesConfig(enabled=True))
    agent = Agent(id="hooked", model="fake-model", instructions="Base instructions", functions=[])
    xerxes.register_agent(agent)

    result = await xerxes.create_response(prompt="say hi", agent_id=agent, stream=False)

    assert result.content == "final"
    persisted_messages = llm.calls[1]["prompt"]
    tool_messages = _tool_messages(persisted_messages)
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "ECHO:HOOKED-HI:AFTER"


@pytest.mark.asyncio
async def test_runtime_policy_override_and_sandbox_routing_apply_at_execution_seam():
    def dangerous() -> str:
        return "ok"

    runtime_features = RuntimeFeaturesConfig(
        enabled=True,
        policy=ToolPolicy(deny={"dangerous"}),
        agent_overrides={
            "allowed": AgentRuntimeOverrides(policy=ToolPolicy(allow={"dangerous"})),
        },
    )
    xerxes = Xerxes(runtime_features=runtime_features)
    allowed = Agent(id="allowed", model="fake", instructions="Allowed", functions=[dangerous])
    denied = Agent(id="denied", model="fake", instructions="Denied", functions=[dangerous])
    xerxes.register_agent(allowed)
    xerxes.register_agent(denied)

    allowed_call = await xerxes.executor._execute_single_call(
        RequestFunctionCall(name="dangerous", arguments={}),
        {},
        allowed,
        runtime_features_state=xerxes._runtime_features_state,
    )
    denied_call = await xerxes.executor._execute_single_call(
        RequestFunctionCall(name="dangerous", arguments={}),
        {},
        denied,
        runtime_features_state=xerxes._runtime_features_state,
    )

    assert allowed_call.status == ExecutionStatus.SUCCESS
    assert denied_call.status == ExecutionStatus.FAILURE
    assert "denied by policy" in (denied_call.error or "")

    strict_xerxes = Xerxes(
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            sandbox=SandboxConfig(mode=SandboxMode.STRICT, sandboxed_tools={"dangerous"}),
        )
    )
    strict_agent = Agent(id="strict", model="fake", instructions="Strict", functions=[dangerous])
    strict_xerxes.register_agent(strict_agent)

    strict_call = await strict_xerxes.executor._execute_single_call(
        RequestFunctionCall(name="dangerous", arguments={}),
        {},
        strict_agent,
        runtime_features_state=strict_xerxes._runtime_features_state,
    )
    assert strict_call.status == ExecutionStatus.FAILURE
    assert "requires sandbox execution" in (strict_call.error or "")

    warn_xerxes = Xerxes(
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            sandbox=SandboxConfig(mode=SandboxMode.WARN, sandboxed_tools={"dangerous"}),
        )
    )
    warn_agent = Agent(id="warn", model="fake", instructions="Warn", functions=[dangerous])
    warn_xerxes.register_agent(warn_agent)

    warn_call = await warn_xerxes.executor._execute_single_call(
        RequestFunctionCall(name="dangerous", arguments={}),
        {},
        warn_agent,
        runtime_features_state=warn_xerxes._runtime_features_state,
    )
    assert warn_call.status == ExecutionStatus.SUCCESS


@pytest.mark.asyncio
async def test_loop_detector_is_reused_across_reinvocation_cycles():
    call_counter = {"count": 0}

    def repeat_tool(query: str) -> str:
        call_counter["count"] += 1
        return query

    llm = _FakeLLM(
        responses=[
            [
                _chunk(
                    function_calls=[{"id": "call_1", "name": "repeat_tool", "arguments": {"query": "same"}}],
                    is_final=True,
                )
            ],
            [
                _chunk(
                    function_calls=[{"id": "call_2", "name": "repeat_tool", "arguments": {"query": "same"}}],
                    is_final=True,
                )
            ],
            [_chunk(content="stopped", is_final=True)],
        ]
    )
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            loop_detection=LoopDetectionConfig(
                same_call_warning=1,
                same_call_critical=2,
                pingpong_warning=10,
                pingpong_critical=20,
                max_tool_calls_per_turn=10,
            ),
        ),
    )
    agent = Agent(id="looping", model="fake-model", instructions="Looping", functions=[repeat_tool])
    xerxes.register_agent(agent)

    result = await xerxes.create_response(prompt="loop", agent_id=agent, stream=False)

    assert result.content == "stopped"
    assert call_counter["count"] == 1
    assert len(llm.calls) == 3
    assert "Tool loop detected" in _tool_messages(llm.calls[2]["prompt"])[0]["content"]


def test_create_llm_resolves_provider_plugins():
    class _ProviderLLM(_FakeLLM):
        def __init__(self, config=None, **kwargs):
            super().__init__(responses=[])
            self.received_config = config
            self.received_kwargs = kwargs

    registry = PluginRegistry()
    registry.register_provider("toy", _ProviderLLM, plugin_name="toy_plugin")

    llm = create_llm("toy", plugin_registry=registry, model="toy-model")

    assert isinstance(llm, _ProviderLLM)


def test_audit_events_emitted_during_live_tool_execution(tmp_path, monkeypatch):
    """Audit events must be emitted from real runtime flow, not only unit tests."""
    from xerxes.audit import InMemoryCollector

    monkeypatch.chdir(tmp_path)
    collector = InMemoryCollector()

    def greet(name: str) -> str:
        return f"Hello {name}"

    llm = _FakeLLM(
        responses=[
            [_chunk(function_calls=[{"id": "c1", "name": "greet", "arguments": {"name": "World"}}], is_final=True)],
            [_chunk(content="done", is_final=True)],
        ]
    )
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            audit_collector=collector,
        ),
    )
    agent = Agent(id="audited", model="fake-model", instructions="Test", functions=[greet])
    xerxes.register_agent(agent)

    import asyncio

    result = asyncio.get_event_loop().run_until_complete(
        xerxes.create_response(prompt="hi", agent_id=agent, stream=False)
    )
    assert result.content == "done"

    events = collector.get_events()
    event_types = [e.event_type for e in events]
    assert "turn_start" in event_types
    assert "tool_call_attempt" in event_types
    assert "tool_call_complete" in event_types
    assert "turn_end" in event_types
    turn_ids = {e.event_type: getattr(e, "turn_id", None) for e in events}
    assert turn_ids["turn_start"] is not None
    assert turn_ids["tool_call_attempt"] == turn_ids["turn_start"]
    assert turn_ids["tool_call_complete"] == turn_ids["turn_start"]
    assert turn_ids["turn_end"] == turn_ids["turn_start"]


def test_session_persistence_during_live_execution(tmp_path, monkeypatch):
    """Session manager must record turns from real runs."""
    from xerxes.session import InMemorySessionStore

    monkeypatch.chdir(tmp_path)
    store = InMemorySessionStore()

    def echo(text: str) -> str:
        return text

    llm = _FakeLLM(
        responses=[
            [_chunk(function_calls=[{"id": "c1", "name": "echo", "arguments": {"text": "ping"}}], is_final=True)],
            [_chunk(content="pong", is_final=True)],
        ]
    )
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            session_store=store,
        ),
    )
    agent = Agent(id="sessioned", model="fake-model", instructions="Test", functions=[echo])
    xerxes.register_agent(agent)

    import asyncio

    result = asyncio.get_event_loop().run_until_complete(
        xerxes.create_response(prompt="test", agent_id=agent, stream=False)
    )
    assert result.content == "pong"

    sessions = store.list_sessions()
    assert len(sessions) == 1
    session = store.load_session(sessions[0])
    assert session is not None
    assert len(session.turns) == 1
    turn = session.turns[0]
    assert turn.prompt == "test"
    assert turn.response_content == "pong"
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].tool_name == "echo"
    assert turn.tool_calls[0].result == "ping"
    assert turn.tool_calls[0].status == "success"


def test_non_tool_runtime_path_emits_turn_end_and_persists_session():
    """Plain completion turns should still emit end events and persist a turn."""
    from xerxes.audit import InMemoryCollector
    from xerxes.session import InMemorySessionStore

    collector = InMemoryCollector()
    store = InMemorySessionStore()
    llm = _FakeLLM(responses=[[_chunk(content="plain answer", is_final=True)]])
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            audit_collector=collector,
            session_store=store,
        ),
    )
    agent = Agent(id="plain_runtime", model="fake-model", instructions="Test", functions=[])
    xerxes.register_agent(agent)

    import asyncio

    result = asyncio.get_event_loop().run_until_complete(
        xerxes.create_response(prompt="hello", agent_id=agent, stream=False, apply_functions=False)
    )
    assert result.content == "plain answer"

    events = collector.get_events()
    event_types = [e.event_type for e in events]
    assert event_types == ["turn_start", "turn_end"]
    assert events[0].turn_id == events[1].turn_id

    sessions = store.list_sessions()
    assert len(sessions) == 1
    session = store.load_session(sessions[0])
    assert session is not None
    assert len(session.turns) == 1
    turn = session.turns[0]
    assert turn.prompt == "hello"
    assert turn.response_content == "plain answer"
    assert turn.tool_calls == []


def test_prompt_profile_compact_mode_produces_shorter_output(tmp_path, monkeypatch):
    """Compact prompt profile must produce a shorter prompt than full."""
    from xerxes.runtime.profiles import PromptProfile

    monkeypatch.chdir(tmp_path)

    xerxes_full = Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=True, guardrails=["Be safe"]))
    agent_full = Agent(id="full", model="fake", instructions="Base", functions=[])
    xerxes_full.register_agent(agent_full)
    full_prompt = xerxes_full.manage_messages(agent=agent_full, prompt="hello").messages[0].content

    xerxes_compact = Xerxes(
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            guardrails=["Be safe"],
            default_prompt_profile=PromptProfile.COMPACT,
        )
    )
    agent_compact = Agent(id="compact", model="fake", instructions="Base", functions=[])
    xerxes_compact.register_agent(agent_compact)
    compact_prompt = xerxes_compact.manage_messages(agent=agent_compact, prompt="hello").messages[0].content

    assert "[Identity]" in full_prompt
    assert "[Tooling]" in full_prompt
    assert "[Runtime Context]" in full_prompt
    assert "[Workspace]" in full_prompt
    # Compact drops workspace
    assert "[Workspace]" not in compact_prompt
    assert "[Safety]" in compact_prompt  # safety preserved


def test_sandbox_backend_instantiation_from_config():
    """Runtime features should instantiate sandbox backend from config backend_type."""
    xerxes = Xerxes(
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            sandbox=SandboxConfig(
                mode=SandboxMode.STRICT,
                sandboxed_tools={"run_code"},
                backend_type="subprocess",
            ),
        )
    )
    state = xerxes._runtime_features_state
    assert state is not None
    assert state.sandbox_backend is not None
    assert state.sandbox_backend.is_available()

    router = state.get_sandbox_router(None)
    assert router is not None
    assert router.backend is not None


def _sandbox_compute(x: int) -> int:
    return x * 2


@pytest.mark.asyncio
async def test_strict_sandbox_with_subprocess_backend_executes():
    """Strict sandbox with subprocess backend should actually execute in sandbox."""
    xerxes = Xerxes(
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            sandbox=SandboxConfig(
                mode=SandboxMode.STRICT,
                sandboxed_tools={"_sandbox_compute"},
                backend_type="subprocess",
            ),
        )
    )
    agent = Agent(id="strict_sb", model="fake", instructions="Test", functions=[_sandbox_compute])
    xerxes.register_agent(agent)

    result = await xerxes.executor._execute_single_call(
        RequestFunctionCall(name="_sandbox_compute", arguments={"x": 5}),
        {},
        agent,
        runtime_features_state=xerxes._runtime_features_state,
    )
    assert result.status == ExecutionStatus.SUCCESS
    assert result.result == 10


def test_plugin_dependency_validation_runs_at_startup(tmp_path, monkeypatch):
    """Plugin dependency validation should surface issues during startup."""
    monkeypatch.chdir(tmp_path)
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    # Plugin with unmet dependency
    plugin_dir.joinpath("needy_plugin.py").write_text(
        """from xerxes.extensions.plugins import PluginMeta, PluginType

PLUGIN_META = PluginMeta(
    name="needy",
    version="1.0.0",
    plugin_type=PluginType.TOOL,
    dependencies=["nonexistent_dep"],
)

def my_tool(x: str) -> str:
    return x

def register(registry):
    registry.register_tool("my_tool", my_tool, meta=PLUGIN_META)
"""
    )

    with pytest.raises(ValueError, match="nonexistent_dep"):
        Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=True))
