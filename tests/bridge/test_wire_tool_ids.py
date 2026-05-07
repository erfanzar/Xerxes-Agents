from __future__ import annotations

import io
import json

from xerxes.bridge.server import BridgeServer


def test_wire_tool_result_reuses_generated_tool_call_id() -> None:
    server = BridgeServer(wire_mode=True)
    output = io.StringIO()
    server._stdout = output

    server._emit_wire_tool_start("", "ExecuteShell", {"command": "cd /tmp && pwd"})
    server._emit_wire_tool_result("", "ok")

    lines = [json.loads(line) for line in output.getvalue().splitlines()]
    tool_call = lines[0]["params"]["payload"]
    tool_result = lines[1]["params"]["payload"]

    assert tool_call["id"]
    assert tool_result["tool_call_id"] == tool_call["id"]
