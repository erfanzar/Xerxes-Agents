from __future__ import annotations

import asyncio
import json
from io import BytesIO
from types import SimpleNamespace

import pytest
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
            },
            None,
        )
    ]
