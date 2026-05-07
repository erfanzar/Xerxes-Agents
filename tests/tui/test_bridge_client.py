from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

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
