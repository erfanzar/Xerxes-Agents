from __future__ import annotations

import asyncio
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from xerxes.core.paths import xerxes_subdir
from xerxes.streaming.wire_events import Notification
from xerxes.tui.engine import BridgeClient


def test_bridge_client_drains_stderr_and_keeps_tail() -> None:
    client = BridgeClient()
    lines = [f"line {idx}\n".encode() for idx in range(205)]
    client._proc = SimpleNamespace(stderr=BytesIO(b"".join(lines)))  # type: ignore[assignment]
    client._running = True

    client._read_stderr_loop()

    tail = client.stderr_tail()
    assert len(tail) == 200
    assert tail[0] == "line 5"
    assert tail[-1] == "line 204"


def test_bridge_client_ready_error_includes_stderr_tail() -> None:
    client = BridgeClient()
    with client._stderr_lock:
        client._stderr_lines.extend(["line 1", "fatal daemon error"])

    message = client._daemon_ready_error()

    assert "did not become ready" in message
    assert "fatal daemon error" in message


def test_bridge_client_defaults_to_project_scoped_daemon_paths(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XERXES_HOME", str(tmp_path / "home"))
    project = tmp_path / "jax-metallib"
    project.mkdir()

    client = BridgeClient(project_dir=project)
    config = client._daemon_config()

    assert config.project_dir == str(project.resolve())
    assert Path(config.socket_path) != xerxes_subdir("daemon", "xerxes.sock")
    assert Path(config.pid_file) != xerxes_subdir("daemon", "daemon.pid")
    assert Path(config.socket_path).parent.name == "projects"
    assert Path(config.pid_file).parent.name == "projects"


def test_bridge_client_project_daemon_paths_are_isolated(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XERXES_HOME", str(tmp_path / "home"))
    first = tmp_path / "jax-metallib"
    second = tmp_path / "xerxes-agents"
    first.mkdir()
    second.mkdir()

    first_config = BridgeClient(project_dir=first)._daemon_config()
    second_config = BridgeClient(project_dir=second)._daemon_config()

    assert first_config.socket_path != second_config.socket_path
    assert first_config.pid_file != second_config.pid_file


def test_bridge_client_honors_explicit_daemon_socket(monkeypatch, tmp_path) -> None:
    socket_path = tmp_path / "explicit.sock"
    monkeypatch.setenv("XERXES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XERXES_DAEMON_SOCKET", str(socket_path))

    config = BridgeClient(project_dir=tmp_path / "project")._daemon_config()

    assert config.socket_path == str(socket_path)


@pytest.mark.asyncio
async def test_bridge_client_preserves_notification_subtype() -> None:
    client = BridgeClient()
    client._loop = asyncio.get_running_loop()
    frame = {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {
            "type": "notification",
            "payload": {
                "id": "resume-list",
                "category": "history",
                "type": "resume_choices",
                "severity": "info",
                "title": "Resume session",
                "body": "Choose a saved session.",
                "payload": {"sessions": [{"session_id": "abcd1234", "title": "first prompt"}]},
            },
        },
    }

    client._handle_inbound_line(json.dumps(frame))
    event = await asyncio.wait_for(client._event_queue.get(), timeout=1)

    assert isinstance(event, Notification)
    assert event.type == "resume_choices"
    assert event.payload["sessions"][0]["session_id"] == "abcd1234"


@pytest.mark.asyncio
async def test_initialize_defaults_to_accept_all_permissions(monkeypatch) -> None:
    client = BridgeClient()
    sent: list[tuple[str, dict, str | None]] = []

    async def send_jsonrpc(*, method: str, params: dict, req_id: str | None = None) -> None:
        sent.append((method, params, req_id))

    monkeypatch.setattr(client, "_send_jsonrpc", send_jsonrpc)

    await client.initialize()

    assert sent == [
        (
            "initialize",
            {
                "model": "",
                "base_url": "",
                "api_key": "",
                "permission_mode": "accept-all",
                "resume_session_id": "",
                "project_dir": client._project_dir,
            },
            None,
        )
    ]
