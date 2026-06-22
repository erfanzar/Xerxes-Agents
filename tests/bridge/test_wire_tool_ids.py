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

from __future__ import annotations

import io
import json

from xerxes.bridge.server import BridgeServer


def test_wire_tool_result_reuses_generated_tool_call_id() -> None:
    server = BridgeServer(wire_mode=True)
    output = io.StringIO()
    server._stdout = output

    server._emit_wire_tool_start("", "ExecuteShell", {"command": "cd /tmp && pwd"})
    server._emit_wire_tool_result("", "ok", duration_ms=123.0)

    lines = [json.loads(line) for line in output.getvalue().splitlines()]
    tool_call = lines[0]["params"]["payload"]
    tool_result = lines[1]["params"]["payload"]

    assert tool_call["id"]
    assert tool_result["tool_call_id"] == tool_call["id"]
    assert tool_result["duration_ms"] == 123.0
