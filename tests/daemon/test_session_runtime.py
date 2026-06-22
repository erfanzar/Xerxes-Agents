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

import pytest
from xerxes.daemon.config import DaemonConfig, load_config
from xerxes.daemon.fingerprint import DAEMON_PROTOCOL_VERSION, daemon_build_id
from xerxes.daemon.runtime import RuntimeManager, SessionManager, TurnRunner, WorkspaceManager
from xerxes.daemon.server import MIGRATED_ERROR, DaemonServer
from xerxes.streaming.events import TextChunk, TurnDone


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

    tool_names = {schema["name"] for schema in runtime.tool_schemas}
    assert {
        "exec_command",
        "write_stdin",
        "list_terminal_sessions",
        "close_terminal_session",
    } <= tool_names

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
