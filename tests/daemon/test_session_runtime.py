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
from xerxes.daemon.runtime import SessionManager, TurnRunner, WorkspaceManager
from xerxes.daemon.server import MIGRATED_ERROR, DaemonServer


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
    runtime = type("Runtime", (), {"model": "", "runtime_config": {}})()
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
    assert server._connection_sessions[emit_first] == "tui:first"
    assert server._connection_sessions[emit_second] == "tui:second"
    assert first_session is not None
    assert second_session is not None
    assert first_session is not second_session
    assert first_session.runtime_config["model"] == "model-a"
    assert second_session.runtime_config["model"] == "model-b"
    assert any(event_type == "init_done" and payload["model"] == "model-a" for event_type, payload in first_events)
    assert any(event_type == "init_done" and payload["model"] == "model-b" for event_type, payload in second_events)


@pytest.mark.asyncio
async def test_workspace_slash_init_creates_project_agents_layout(tmp_path):
    cfg = DaemonConfig(
        workspace={"root": str(tmp_path / "agents"), "default_agent_id": "xerxes"},
        project_dir=str(tmp_path),
    )
    server = DaemonServer(cfg)
    session = server.sessions.open("tui:default", "xerxes")
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> None:
        events.append((event_type, payload))

    await server._slash_workspace("init", emit)

    assert (tmp_path / ".agents" / "AGENTS.md").is_file()
    assert (tmp_path / ".agents" / "skills").is_dir()
    assert session.project_dir == tmp_path.resolve()
    body = events[-1][1]["body"]
    assert "Project .agents" in body
    assert "Loaded project context" in body


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
