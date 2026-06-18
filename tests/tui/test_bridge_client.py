from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import pytest
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
