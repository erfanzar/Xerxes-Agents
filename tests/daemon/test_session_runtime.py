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
