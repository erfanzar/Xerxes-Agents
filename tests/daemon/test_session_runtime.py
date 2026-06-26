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

import asyncio
import json
import queue
import tempfile
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from xerxes.channels import ChannelMessage, MessageDirection
from xerxes.daemon.config import DaemonConfig, load_config
from xerxes.daemon.fingerprint import DAEMON_PROTOCOL_VERSION, daemon_build_id
from xerxes.daemon.runtime import RuntimeManager, SessionManager, TurnRunner, WorkspaceManager
from xerxes.daemon.server import MIGRATED_ERROR, DaemonServer
from xerxes.streaming.events import ProviderRetry, TextChunk, TurnDone


def test_config_env_refs_and_legacy_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-secret")
    monkeypatch.setenv("XERXES_MODEL", "env-model")
    monkeypatch.setenv("XERXES_DAEMON_ENABLE_TELEGRAM", "1")

    cfg = load_config(project_dir="/tmp/project")

    assert cfg.project_dir == "/tmp/project"
    assert cfg.model == "env-model"
    telegram = cfg.resolved_channels()["telegram"]
    assert telegram["enabled"] is True
    assert telegram["settings"]["token"] == "telegram-secret"


def test_config_discord_gateway_env(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "discord-secret")
    monkeypatch.setenv("XERXES_DAEMON_ENABLE_DISCORD", "1")
    monkeypatch.setenv("XERXES_DISCORD_CHANNEL_NAME", "m2-max")
    monkeypatch.setenv("XERXES_DISCORD_INSTANCE_NAME", "mac-studio")
    monkeypatch.setenv("XERXES_DISCORD_ADDRESS_NAME", "mac")

    cfg = load_config(project_dir="/tmp/project")

    discord = cfg.resolved_channels()["discord"]
    assert discord["enabled"] is True
    assert discord["type"] == "discord"
    assert discord["settings"]["token"] == "discord-secret"
    assert discord["settings"]["transport"] == "gateway"
    assert discord["settings"]["require_mention"] is True
    assert discord["settings"]["allowed_channel_names"] == "m2-max"
    assert discord["settings"]["instance_name"] == "mac-studio"
    assert discord["settings"]["address_names"] == "mac"


class FakeChannelInstance:
    def __init__(self) -> None:
        self.sent: list[ChannelMessage] = []
        self.typing_rooms: list[str | None] = []

    async def send(self, message: ChannelMessage) -> None:
        self.sent.append(message)

    async def send_typing(self, room_id: str | None) -> None:
        self.typing_rooms.append(room_id)


def _discord_message(text: str) -> ChannelMessage:
    return ChannelMessage(
        text=text,
        channel="discord",
        channel_user_id="U1",
        room_id="C1",
        platform_message_id="M1",
        direction=MessageDirection.INBOUND,
        metadata={"chat_type": "group", "thread_id": "main"},
    )


@pytest.mark.asyncio
async def test_channel_slash_skills_replies_without_model_turn(tmp_path):
    cfg = DaemonConfig(
        workspace={"root": str(tmp_path / "agents"), "default_agent_id": "xerxes"},
        project_dir=str(tmp_path),
    )
    server = DaemonServer(cfg)
    channel = FakeChannelInstance()
    server.channels.channels["discord"] = SimpleNamespace(instance=channel)
    server.runtime.skills_list_text = lambda: "Skills:\n  /deep"  # type: ignore[method-assign]
    server.runtime.discover_skills = lambda: ["deep"]  # type: ignore[method-assign]

    await server._handle_channel_message(_discord_message("/skills"))

    assert len(channel.sent) == 1
    assert channel.sent[0].text == "Skills:\n  /deep"
    assert channel.sent[0].reply_to == "M1"


@pytest.mark.asyncio
async def test_channel_slash_ask_runs_turn_and_refreshes_typing(tmp_path):
    cfg = DaemonConfig(
        workspace={"root": str(tmp_path / "agents"), "default_agent_id": "xerxes"},
        project_dir=str(tmp_path),
    )
    server = DaemonServer(cfg)
    channel = FakeChannelInstance()
    server.channels.channels["discord"] = SimpleNamespace(instance=channel)
    captured_prompts: list[str] = []

    async def run_turn(session, text, emit, *, mode: str = "code", plan_mode: bool = False):
        captured_prompts.append(text)
        await asyncio.sleep(0.01)
        await emit("text_part", {"text": "answer"})
        return "answer"

    server.turns.run_turn = run_turn  # type: ignore[method-assign]

    await server._handle_channel_message(_discord_message("/ask which dir?"))

    assert captured_prompts == ["[discord message]\nroom_id: C1\nfrom_user_id: U1\nthread_id: main\n\nwhich dir?"]
    assert channel.typing_rooms == ["C1"]
    assert len(channel.sent) == 1
    assert channel.sent[0].text == "answer"


def test_channel_skill_prompt_parses_name_and_prompt_argument(tmp_path):
    cfg = DaemonConfig(
        workspace={"root": str(tmp_path / "agents"), "default_agent_id": "xerxes"},
        project_dir=str(tmp_path),
    )
    server = DaemonServer(cfg)
    server.runtime.skills_dir = tmp_path / "skills"
    skill_dir = server.runtime.skills_dir / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo skill\n---\n# Procedure\nDo the thing.\n",
        encoding="utf-8",
    )

    result = server._channel_skill_prompt("skill", "demo inspect the repo")

    assert result is not None
    name, prompt, error = result
    assert name == "demo"
    assert error == ""
    assert "## Skill: demo" in prompt
    assert "User request: inspect the repo" in prompt


def test_session_manager_creates_workspace_under_agents_root(tmp_path):
    cfg = DaemonConfig(workspace={"root": str(tmp_path / "agents"), "default_agent_id": "xerxes"})
    manager = SessionManager(WorkspaceManager(cfg), keep_messages=4)

    session = manager.open("telegram:private:1")

    assert session.key == "telegram:private:1"
    assert session.workspace.path == tmp_path / "agents" / "xerxes"
    assert (session.workspace.path / "AGENTS.md").exists()
    assert (session.workspace.path / "SOUL.md").exists()
    assert (session.workspace.path / "MEMORY.md").exists()
    assert (session.workspace.path / "memory").is_dir()


def test_session_save_records_project_cwd_not_agent_workspace(tmp_path):
    project = tmp_path / "project"
    cfg = DaemonConfig(
        workspace={"root": str(tmp_path / "agents"), "default_agent_id": "xerxes"},
        project_dir=str(project),
    )
    manager = SessionManager(WorkspaceManager(cfg), keep_messages=4, store_dir=tmp_path / "sessions")
    session = manager.open("tui:default")
    session.state.messages = [{"role": "user", "content": "hello"}]

    manager.save(session)

    record = json.loads((tmp_path / "sessions" / f"{session.id}.json").read_text(encoding="utf-8"))
    assert record["cwd"] == str(project.resolve())
    assert record["workspace"] == str(tmp_path / "agents" / "xerxes")
    assert record["cwd"] != record["workspace"]


def test_session_load_migrates_old_workspace_cwd_to_current_project(tmp_path):
    project = tmp_path / "project"
    workspace_root = tmp_path / "agents"
    store_dir = tmp_path / "sessions"
    store_dir.mkdir()
    old_workspace = workspace_root / "xerxes"
    old_workspace.mkdir(parents=True)
    cfg = DaemonConfig(
        workspace={"root": str(workspace_root), "default_agent_id": "xerxes"},
        project_dir=str(project),
    )
    record = {
        "session_id": "abcd1234",
        "key": "abcd1234",
        "agent_id": "xerxes",
        "cwd": str(old_workspace),
        "messages": [{"role": "user", "content": "hello"}],
        "turn_count": 1,
    }
    (store_dir / "abcd1234.json").write_text(json.dumps(record), encoding="utf-8")

    session = SessionManager(WorkspaceManager(cfg), keep_messages=4, store_dir=store_dir).open("abcd1234")

    assert session.project_dir == project.resolve()


def test_status_payload_uses_session_scoped_interaction_mode(tmp_path):
    cfg = DaemonConfig(
        workspace={"root": str(tmp_path / "agents"), "default_agent_id": "xerxes"},
        project_dir=str(tmp_path),
    )
    sessions = SessionManager(WorkspaceManager(cfg), keep_messages=4)
    session = sessions.open("tui:default")
    runtime = type(
        "Runtime",
        (),
        {
            "model": "",
            "runtime_config": {},
            "system_prompt": "",
            "tool_schemas": [],
            "active_skill_prompt": lambda self: "",
        },
    )()
    runner = TurnRunner(runtime=runtime, sessions=sessions)
    try:
        runner._set_session_mode(session, "researcher")

        payload = runner._status_payload(session, mode="code", plan_mode=False)

        assert payload["mode"] == "researcher"
        assert payload["plan_mode"] is False
        assert session.status()["mode"] == "researcher"
    finally:
        runner.close()


def test_status_payload_uses_session_scoped_reasoning_effort(tmp_path):
    cfg = DaemonConfig(
        workspace={"root": str(tmp_path / "agents"), "default_agent_id": "xerxes"},
        project_dir=str(tmp_path),
    )
    sessions = SessionManager(WorkspaceManager(cfg), keep_messages=4)
    session = sessions.open("tui:default")
    session.runtime_config = {"model": "claude-code/opus", "thinking": True, "reasoning_effort": "high"}
    runtime = type(
        "Runtime",
        (),
        {
            "model": "",
            "runtime_config": {},
            "system_prompt": "",
            "tool_schemas": [],
            "active_skill_prompt": lambda self: "",
        },
    )()
    runner = TurnRunner(runtime=runtime, sessions=sessions)
    try:
        payload = runner._status_payload(session, mode="code", plan_mode=False)

        assert payload["reasoning_effort"] == "high"

        session.runtime_config = {"model": "claude-code/opus", "thinking": False, "reasoning_effort": "high"}
        payload = runner._status_payload(session, mode="code", plan_mode=False)

        assert payload["reasoning_effort"] == "off"
    finally:
        runner.close()


def test_claude_code_runtime_defaults_to_medium_reasoning(tmp_path):
    runtime = RuntimeManager(DaemonConfig(project_dir=str(tmp_path)))
    config = {"provider": "claude-code", "model": "claude-code/opus"}

    runtime._normalize_reasoning_config(config)

    assert config["thinking"] is True
    assert config["reasoning_effort"] == "medium"
    assert config["thinking_budget"] == runtime.REASONING_LEVELS["medium"]


def test_reasoning_effort_override_enables_thinking(tmp_path):
    runtime = RuntimeManager(DaemonConfig(project_dir=str(tmp_path)))
    config = {"provider": "zhipu", "model": "glm-5.2", "reasoning_effort": "high"}

    runtime._normalize_reasoning_config(config)

    assert config["thinking"] is True
    assert config["reasoning_effort"] == "high"
    assert config["thinking_budget"] == runtime.REASONING_LEVELS["high"]


def test_explicit_thinking_false_wins_over_stale_effort(tmp_path):
    runtime = RuntimeManager(DaemonConfig(project_dir=str(tmp_path)))
    config = {"provider": "claude-code", "model": "claude-code/opus", "thinking": False, "reasoning_effort": "high"}

    runtime._normalize_reasoning_config(config)

    assert config["thinking"] is False
    assert config["thinking_budget"] == 0
    assert "reasoning_effort" not in config


def test_session_mode_does_not_sync_from_process_global_config(tmp_path):
    cfg = DaemonConfig(
        workspace={"root": str(tmp_path / "agents"), "default_agent_id": "xerxes"},
        project_dir=str(tmp_path),
    )
    sessions = SessionManager(WorkspaceManager(cfg), keep_messages=4)
    session = sessions.open("tui:default")
    runtime = type("Runtime", (), {"model": "", "runtime_config": {}})()
    runner = TurnRunner(runtime=runtime, sessions=sessions)
    try:
        runner._set_session_mode(session, "researcher", publish=True)

        runner._sync_session_mode_from_global(session)

        assert session.interaction_mode == "researcher"
        assert session.plan_mode is False
    finally:
        runner.close()


def test_turn_runner_flushes_pending_steers_before_marking_turn_idle(monkeypatch, tmp_path):
    cfg = DaemonConfig(
        workspace={"root": str(tmp_path / "agents"), "default_agent_id": "xerxes"},
        project_dir=str(tmp_path),
    )
    sessions = SessionManager(WorkspaceManager(cfg), keep_messages=4, store_dir=tmp_path / "sessions")
    session = sessions.open("tui:default")
    runtime = RuntimeManager(cfg)
    runtime.runtime_config = {"model": "openai/test", "permission_mode": "accept-all", "project_dir": str(tmp_path)}
    runtime.system_prompt = ""
    runtime.tool_executor = lambda name, inp: ""
    runtime.tool_schemas = []
    runtime.agent_memory = type("Memory", (), {"to_prompt_section": lambda self: ""})()
    runner = TurnRunner(runtime=runtime, sessions=sessions)
    session.pending_steers.put("make a todo for it")
    events: list[tuple[str, dict]] = []

    def fake_run_agent_loop(*args, **kwargs):
        yield TextChunk("done")
        yield TurnDone(input_tokens=1, output_tokens=1, tool_calls_count=0, model="openai/test")

    async def emit(event_type: str, payload: dict) -> None:
        events.append((event_type, payload))

    monkeypatch.setattr("xerxes.daemon.runtime.run_agent_loop", fake_run_agent_loop)
    try:
        asyncio.run(runner.run_turn(session, "finish", emit))
    finally:
        runner.close()

    assert session.active_turn_id == ""
    assert session.drain_steers() == []
    assert any("make a todo for it" in msg.get("content", "") for msg in session.state.messages)
    assert any(
        event_type == "notification" and "Steer saved for next turn" in payload.get("body", "")
        for event_type, payload in events
    )


def test_turn_runner_emits_provider_retry_as_notification_not_text(monkeypatch, tmp_path):
    cfg = DaemonConfig(
        workspace={"root": str(tmp_path / "agents"), "default_agent_id": "xerxes"},
        project_dir=str(tmp_path),
    )
    sessions = SessionManager(WorkspaceManager(cfg), keep_messages=4, store_dir=tmp_path / "sessions")
    session = sessions.open("tui:default")
    runtime = RuntimeManager(cfg)
    runtime.runtime_config = {"model": "openai/test", "permission_mode": "accept-all", "project_dir": str(tmp_path)}
    runtime.system_prompt = ""
    runtime.tool_executor = lambda name, inp: ""
    runtime.tool_schemas = []
    runtime.agent_memory = type("Memory", (), {"to_prompt_section": lambda self: ""})()
    runner = TurnRunner(runtime=runtime, sessions=sessions)
    events: list[tuple[str, dict]] = []

    def fake_run_agent_loop(*args, **kwargs):
        yield ProviderRetry(error="Request timed out.", attempt=2, max_attempts=6, delay=5)
        yield TurnDone(input_tokens=0, output_tokens=0, tool_calls_count=0, model="openai/test")

    async def emit(event_type: str, payload: dict) -> None:
        events.append((event_type, payload))

    monkeypatch.setattr("xerxes.daemon.runtime.run_agent_loop", fake_run_agent_loop)
    try:
        output = asyncio.run(runner.run_turn(session, "hello", emit))
    finally:
        runner.close()

    assert output == ""
    assert not any(event_type == "text_part" for event_type, _payload in events)
    retry_notifications = [
        payload
        for event_type, payload in events
        if event_type == "notification" and payload.get("category") == "provider_connection"
    ]
    assert len(retry_notifications) == 1
    assert retry_notifications[0]["type"] == "retrying"
    assert "Retrying provider connection in 5s" in retry_notifications[0]["body"]


def test_daemon_runtime_event_updates_current_mode(tmp_path):
    cfg = DaemonConfig(
        workspace={"root": str(tmp_path / "agents"), "default_agent_id": "xerxes"},
        project_dir=str(tmp_path),
    )
    server = DaemonServer(cfg)
    server._current_mode = "researcher"
    server._current_plan_mode = False

    server._handle_runtime_event("interaction_mode_changed", {"mode": "code", "plan_mode": False})

    assert server._current_mode == "code"
    assert server._current_plan_mode is False
    assert server.runtime.runtime_config["mode"] == "code"


def test_daemon_runtime_event_updates_only_named_session(tmp_path):
    cfg = DaemonConfig(
        workspace={"root": str(tmp_path / "agents"), "default_agent_id": "xerxes"},
        project_dir=str(tmp_path),
    )
    server = DaemonServer(cfg)
    first = server.sessions.open("tui:first", "xerxes")
    second = server.sessions.open("tui:second", "xerxes")
    first.interaction_mode = "researcher"
    second.interaction_mode = "code"

    server._handle_runtime_event(
        "interaction_mode_changed",
        {"mode": "objective", "plan_mode": False, "session_key": "tui:first"},
    )

    assert first.interaction_mode == "objective"
    assert second.interaction_mode == "code"


@pytest.mark.asyncio
async def test_initialize_uses_connection_session_key_and_runtime_snapshot(tmp_path):
    cfg = DaemonConfig(
        workspace={"root": str(tmp_path / "agents"), "default_agent_id": "xerxes"},
        project_dir=str(tmp_path),
    )
    server = DaemonServer(cfg)
    server.runtime.discover_skills = lambda: []
    server._git_branch = lambda cwd=None: ""

    def reload_runtime(overrides=None):
        runtime = {"permission_mode": "accept-all", "model": ""}
        runtime.update(overrides or {})
        server.runtime.runtime_config = runtime

    server.runtime.reload = reload_runtime  # type: ignore[method-assign]
    first_events: list[tuple[str, dict]] = []
    second_events: list[tuple[str, dict]] = []

    async def emit_first(event_type: str, payload: dict) -> None:
        first_events.append((event_type, payload))

    async def emit_second(event_type: str, payload: dict) -> None:
        second_events.append((event_type, payload))

    first = await server._initialize({"session_key": "tui:first", "model": "model-a"}, emit_first)
    second = await server._initialize({"session_key": "tui:second", "model": "model-b"}, emit_second)

    first_session = server.sessions.get("tui:first")
    second_session = server.sessions.get("tui:second")
    assert first["ok"] is True
    assert second["ok"] is True
    assert first["daemon_protocol"] == DAEMON_PROTOCOL_VERSION
    assert first["daemon_build_id"] == daemon_build_id()
    assert server._connection_sessions[emit_first] == "tui:first"
    assert server._connection_sessions[emit_second] == "tui:second"
    assert first_session is not None
    assert second_session is not None
    assert first_session is not second_session
    assert first_session.runtime_config["model"] == "model-a"
    assert second_session.runtime_config["model"] == "model-b"
    assert any(event_type == "init_done" and payload["model"] == "model-a" for event_type, payload in first_events)
    assert any(event_type == "init_done" and payload["model"] == "model-b" for event_type, payload in second_events)


def test_runtime_reload_registers_terminal_operator_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("XERXES_HOME", str(tmp_path / "home"))
    runtime = RuntimeManager(DaemonConfig(project_dir=str(tmp_path)))

    runtime.reload({"model": "gpt-4o"})

    schemas_by_name = {schema["name"]: schema for schema in runtime.tool_schemas}
    tool_names = set(schemas_by_name)
    assert {
        "exec_command",
        "write_stdin",
        "list_terminal_sessions",
        "close_terminal_session",
    } <= tool_names

    exec_schema = schemas_by_name["exec_command"]["input_schema"]
    assert exec_schema["properties"]["cmd"]["type"] == "string"
    assert exec_schema["properties"]["yield_time_ms"]["type"] == "integer"
    assert exec_schema["properties"]["login"]["type"] == "boolean"
    assert "cmd" in exec_schema["required"]
    assert "session_id" in schemas_by_name["write_stdin"]["input_schema"]["required"]

    missing_result = runtime.tool_executor("exec_command", {})
    assert missing_result == "Error: exec_command: missing required parameter(s): cmd"
    assert "OperatorState" not in missing_result

    result = runtime.tool_executor("exec_command", {"cmd": "printf hello", "yield_time_ms": 200})

    assert "hello" in result
    assert "session_id" in result


@pytest.mark.asyncio
async def test_workspace_slash_init_queues_agent_driven_project_setup(tmp_path):
    cfg = DaemonConfig(
        workspace={"root": str(tmp_path / "agents"), "default_agent_id": "xerxes"},
        project_dir=str(tmp_path),
    )
    server = DaemonServer(cfg)
    session = server.sessions.open("tui:default", "xerxes")
    events: list[tuple[str, dict]] = []
    captured: list[dict] = []

    async def emit(event_type: str, payload: dict) -> None:
        events.append((event_type, payload))

    async def submit_turn(params: dict, emit_fn) -> dict:
        captured.append(params)
        return {"ok": True}

    server._submit_turn = submit_turn  # type: ignore[method-assign]
    server.runtime.discover_skills = lambda: []  # type: ignore[method-assign]

    await server._slash_workspace("init", emit)

    assert captured
    prompt = captured[0]["text"]
    assert captured[0]["_internal_slash"] is True
    assert "Initialize this repository for Xerxes by running a swarm-backed project discovery." in prompt
    assert f"Project root: `{tmp_path.resolve()}`" in prompt
    assert "Do not use a generic template" in prompt
    assert (
        "Before writing `XERXES.md` or `.agents/` files, spawn parallel discovery subagents with `SpawnAgents`."
        in prompt
    )
    assert "Do not cap the swarm with an arbitrary number." in prompt
    assert "Use `AwaitAgents`, `TaskGetTool`, or `TaskOutputTool` to collect results." in prompt
    assert "Write or update `XERXES.md` only after the swarm findings are synthesized." in prompt
    assert ".agents/skills/<skill-name>/SKILL.md" in prompt
    assert not (tmp_path / "XERXES.md").exists()
    assert session.project_dir == tmp_path.resolve()
    body = [payload["body"] for event_type, payload in events if event_type == "notification"][-1]
    assert "Project initialization turn finished" in body


@pytest.mark.asyncio
async def test_init_slash_submits_project_specific_setup_prompt(tmp_path):
    cfg = DaemonConfig(
        workspace={"root": str(tmp_path / "agents"), "default_agent_id": "xerxes"},
        project_dir=str(tmp_path),
    )
    server = DaemonServer(cfg)
    server.sessions.open("tui:default", "xerxes")
    events: list[tuple[str, dict]] = []
    captured: list[dict] = []

    async def emit(event_type: str, payload: dict) -> None:
        events.append((event_type, payload))

    async def submit_turn(params: dict, emit_fn) -> dict:
        captured.append(params)
        return {"ok": True}

    server._submit_turn = submit_turn  # type: ignore[method-assign]
    server.runtime.discover_skills = lambda: ["project-skill"]  # type: ignore[method-assign]

    await server._slash_init("focus on CI and local skills", emit)

    assert captured
    prompt = captured[0]["text"]
    assert "User request for this init: focus on CI and local skills" in prompt
    assert "existing `AGENTS.md`, existing `XERXES.md`, existing `.agents/`" in prompt
    assert "If subagent tools are unavailable, stop and report" in prompt
    assert "Write or update `XERXES.md` only after the swarm findings are synthesized." in prompt
    assert "not placeholder skills" in prompt
    assert not (tmp_path / "XERXES.md").exists()
    assert any(event_type == "init_done" for event_type, _ in events)
    bodies = [payload["body"] for event_type, payload in events if event_type == "notification"]
    assert any("Project initialization swarm queued" in body for body in bodies)
    assert bodies[-1] == "Project initialization turn finished. Reloaded 1 skill(s)."


@pytest.mark.asyncio
async def test_turn_runner_permission_response_unblocks_request(tmp_path):
    cfg = DaemonConfig(workspace={"root": str(tmp_path / "agents"), "default_agent_id": "xerxes"})
    sessions = SessionManager(WorkspaceManager(cfg), keep_messages=4)
    session = sessions.open("tui:permission")
    runner = TurnRunner(runtime=object(), sessions=sessions)
    request_id = "req-1"
    waiter: queue.Queue[str] = queue.Queue()
    results: list[str] = []
    with runner._permission_lock:
        runner._permission_waiters[request_id] = waiter

    thread = threading.Thread(
        target=lambda: results.append(runner._wait_for_permission_response(session, request_id, waiter))
    )
    thread.start()
    try:
        assert await runner.respond_permission(request_id, "approve")
        thread.join(timeout=1)
        assert results == ["approve"]
    finally:
        runner.close()


def test_turn_runner_emits_subagent_preview_and_nested_tool_event(tmp_path):
    cfg = DaemonConfig(workspace={"root": str(tmp_path / "agents"), "default_agent_id": "xerxes"})
    sessions = SessionManager(WorkspaceManager(cfg), keep_messages=4)
    runner = TurnRunner(runtime=object(), sessions=sessions)
    emitted: list[tuple[str, dict]] = []

    def push(event_type: str, payload: dict) -> None:
        emitted.append((event_type, payload))

    try:
        runner.set_event_sink(push)
        runner._current_tool_call_id = "parent-tool"
        runner.handle_agent_event(
            "agent_spawn",
            {
                "task_id": "subagent-123456",
                "agent_name": "structure-analyzer",
                "agent_type": "researcher",
                "prompt": "Analyze project structure.",
            },
        )
        runner.handle_agent_event(
            "agent_tool_start",
            {
                "task_id": "subagent-123456",
                "agent_type": "researcher",
                "tool_call_id": "inner-tool",
                "tool_name": "ReadFile",
                "inputs": {"path": "README.md"},
            },
        )
    finally:
        runner.close()

    assert emitted[0][0] == "notification"
    assert emitted[0][1]["category"] == "subagent_stream"
    assert emitted[0][1]["payload"]["task_id"] == "subagent-123456"
    assert ("subagent-123456" in runner._subagent_parent_tool) is True
    subagent_events = [payload for event_type, payload in emitted if event_type == "subagent_event"]
    assert subagent_events
    assert subagent_events[0]["parent_tool_call_id"] == "parent-tool"
    assert subagent_events[0]["event"]["payload"]["name"] == "ReadFile"


@pytest.mark.asyncio
async def test_daemon_socket_initialize_and_old_task_error(tmp_path):
    short_tmp = Path(tempfile.gettempdir()) / f"xerxes-d-{uuid.uuid4().hex[:8]}"
    short_tmp.mkdir()
    cfg = DaemonConfig(
        runtime={"model": "test-model", "permission_mode": "accept-all"},
        control={
            "websocket_host": "127.0.0.1",
            "websocket_port": 0,
            "unix_socket": str(short_tmp / "d.sock"),
            "pid_file": str(short_tmp / "daemon.pid"),
            "log_dir": str(short_tmp / "logs"),
        },
        workspace={"root": str(tmp_path / "agents"), "default_agent_id": "default"},
        project_dir=str(tmp_path),
    )
    server = DaemonServer(cfg)
    task = asyncio.create_task(server.run())
    try:
        for _ in range(100):
            if Path(cfg.socket_path).exists():
                break
            await asyncio.sleep(0.05)

        reader, writer = await asyncio.open_unix_connection(cfg.socket_path)
        writer.write(json.dumps({"jsonrpc": "2.0", "id": "1", "method": "initialize", "params": {}}).encode() + b"\n")
        await writer.drain()

        events = []
        result = None
        while result is None:
            msg = json.loads((await asyncio.wait_for(reader.readline(), timeout=5)).decode())
            if msg.get("method") == "event":
                events.append(msg["params"]["type"])
            elif msg.get("id") == "1":
                result = msg["result"]

        assert result["ok"] is True
        assert result["model"] == "test-model"
        assert "init_done" in events

        writer.write(
            json.dumps({"jsonrpc": "2.0", "id": "2", "method": "task.submit", "params": {"prompt": "hi"}}).encode()
            + b"\n"
        )
        await writer.drain()
        migrated = json.loads((await asyncio.wait_for(reader.readline(), timeout=5)).decode())["result"]
        assert migrated == {"ok": False, "error": MIGRATED_ERROR}

        writer.write(json.dumps({"jsonrpc": "2.0", "id": "3", "method": "shutdown", "params": {}}).encode() + b"\n")
        await writer.drain()
        await asyncio.wait_for(reader.readline(), timeout=5)
        writer.close()
        await writer.wait_closed()
    finally:
        if not task.done():
            await server.shutdown()
        await asyncio.wait_for(task, timeout=5)


@pytest.mark.asyncio
async def test_daemon_socket_starts_without_configured_model(tmp_path, monkeypatch):
    monkeypatch.setattr("xerxes.daemon.runtime.profiles.get_active_profile", lambda: None)
    short_tmp = Path(tempfile.gettempdir()) / f"xerxes-d-{uuid.uuid4().hex[:8]}"
    short_tmp.mkdir()
    cfg = DaemonConfig(
        runtime={"permission_mode": "accept-all"},
        control={
            "websocket_host": "127.0.0.1",
            "websocket_port": 0,
            "unix_socket": str(short_tmp / "d.sock"),
            "pid_file": str(short_tmp / "daemon.pid"),
            "log_dir": str(short_tmp / "logs"),
        },
        workspace={"root": str(tmp_path / "agents"), "default_agent_id": "default"},
        project_dir=str(tmp_path),
    )
    server = DaemonServer(cfg)
    task = asyncio.create_task(server.run())
    try:
        for _ in range(100):
            if Path(cfg.socket_path).exists():
                break
            await asyncio.sleep(0.05)

        reader, writer = await asyncio.open_unix_connection(cfg.socket_path)
        writer.write(json.dumps({"jsonrpc": "2.0", "id": "1", "method": "initialize", "params": {}}).encode() + b"\n")
        await writer.drain()

        result = None
        while result is None:
            msg = json.loads((await asyncio.wait_for(reader.readline(), timeout=5)).decode())
            if msg.get("id") == "1":
                result = msg["result"]

        assert result["ok"] is True
        assert result["model"] == ""

        writer.write(
            json.dumps({"jsonrpc": "2.0", "id": "2", "method": "prompt", "params": {"user_input": "hi"}}).encode()
            + b"\n"
        )
        await writer.drain()
        rejected = json.loads((await asyncio.wait_for(reader.readline(), timeout=5)).decode())["result"]
        assert rejected == {"ok": False, "error": "No model configured. Run /provider first or set XERXES_MODEL."}

        writer.write(json.dumps({"jsonrpc": "2.0", "id": "3", "method": "shutdown", "params": {}}).encode() + b"\n")
        await writer.drain()
        await asyncio.wait_for(reader.readline(), timeout=5)
        writer.close()
        await writer.wait_closed()
    finally:
        if not task.done():
            await server.shutdown()
        await asyncio.wait_for(task, timeout=5)
