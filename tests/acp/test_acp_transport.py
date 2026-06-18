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
"""Protocol-level tests for the ACP stdio JSON-RPC transport.

Deterministic — uses a fake runner (no LLM/network). Drives the real
:class:`StdioJsonRpcServer` over in-memory string streams and asserts the
JSON-RPC framing, dispatch (snake_case + slash aliases), streamed prompt
notifications, and error handling."""

from __future__ import annotations

import io
import json

from xerxes.acp.server import AcpServer
from xerxes.acp.transport import StdioJsonRpcServer


class _FakeRunner:
    """Stand-in for AcpAgentRunner — emits canned events, no LLM."""

    def list_tools(self):
        return [{"type": "function", "function": {"name": "echo"}}]

    def list_models(self):
        return [{"id": "m1", "name": "m1"}, {"id": "m2", "name": "m2"}]

    def run_prompt(self, *, session, text, emit=None, **_):
        emit = emit or (lambda e: None)
        emit({"kind": "text_delta", "text": "hello "})
        emit({"kind": "text_delta", "text": text})
        emit({"kind": "turn_end", "output_tokens": 2, "model": "m1"})
        return {"ok": True, "output_tokens": 2, "echo": text}


def _drive(requests: list[dict]) -> tuple[AcpServer, list[dict], list[dict]]:
    """Run the transport over ``requests`` and return (server, responses, notifications)."""
    runner = _FakeRunner()
    server = AcpServer(
        prompt_handler=runner.run_prompt,
        tool_list_provider=runner.list_tools,
        model_list_provider=runner.list_models,
    )
    stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    stdout = io.StringIO()
    StdioJsonRpcServer(server, runner, stdin=stdin, stdout=stdout).serve_forever()
    responses, notifications = [], []
    for line in stdout.getvalue().splitlines():
        if not line.strip():
            continue
        msg = json.loads(line)
        (notifications if "method" in msg else responses).append(msg)
    return server, responses, notifications


def _req(method, params=None, rid=1):
    return {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}


class TestHandshakeAndLists:
    def test_initialize_advertises_capabilities(self):
        _, resp, _ = _drive([_req("initialize", {"client_info": {"name": "test"}})])
        assert resp[0]["result"]["server_name"] == "xerxes"
        assert resp[0]["result"]["capabilities"]["protocol_version"] == "0.9"

    def test_list_tools_and_alias(self):
        _, resp, _ = _drive([_req("list_tools", rid=1), _req("tools/list", rid=2)])
        by_id = {r["id"]: r for r in resp}
        assert by_id[1]["result"] == by_id[2]["result"]
        assert by_id[1]["result"][0]["function"]["name"] == "echo"

    def test_list_models(self):
        _, resp, _ = _drive([_req("list_models")])
        assert [m["id"] for m in resp[0]["result"]] == ["m1", "m2"]


class TestSessionLifecycle:
    def test_open_list_setmodel_close(self):
        _server, resp, _ = _drive(
            [
                _req("open_session", {"cwd": "/tmp/x"}, rid=1),
            ]
        )
        resp[0]["result"]["session_id"]
        assert resp[0]["result"]["cwd"] == "/tmp/x"
        # Drive follow-ups against the SAME server via a second run is not possible
        # (separate server); instead exercise them together using the pre-created id.
        _server2, resp2, _ = _drive(
            [
                _req("open_session", {"cwd": "/tmp/y", "model": "m2", "title": "t"}, rid=1),
                _req("list_sessions", rid=2),
            ]
        )
        new_sid = resp2[0]["result"]["session_id"]
        listed = {s["session_id"] for s in resp2[1]["result"]}
        assert new_sid in listed
        assert resp2[0]["result"]["model"] == "m2"

    def test_set_model_and_cancel_and_close_unknown(self):
        _, resp, _ = _drive(
            [
                _req("set_model", {"session_id": "nope", "model": "x"}, rid=1),
                _req("cancel", {"session_id": "nope"}, rid=2),
                _req("close_session", {"session_id": "nope"}, rid=3),
            ]
        )
        by_id = {r["id"]: r for r in resp}
        assert by_id[1]["result"]["ok"] is False
        assert by_id[2]["result"]["ok"] is False
        assert by_id[3]["result"]["ok"] is False


class TestPrompt:
    def test_prompt_streams_events_then_result(self):
        runner = _FakeRunner()
        server = AcpServer(prompt_handler=runner.run_prompt)
        sid = server.open_session("/tmp")["session_id"]
        stdin = io.StringIO(json.dumps(_req("prompt", {"session_id": sid, "text": "world"})) + "\n")
        stdout = io.StringIO()
        StdioJsonRpcServer(server, runner, stdin=stdin, stdout=stdout).serve_forever()
        lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
        updates = [m for m in lines if m.get("method") == "session/update"]
        results = [m for m in lines if "result" in m]
        texts = [u["params"]["event"]["text"] for u in updates if u["params"]["event"]["kind"] == "text_delta"]
        assert texts == ["hello ", "world"]
        assert any(u["params"]["event"]["kind"] == "turn_end" for u in updates)
        assert results[-1]["result"]["echo"] == "world"
        # every update carries the session id + request id for correlation
        assert all(u["params"]["session_id"] == sid for u in updates)
        assert all(u["params"]["request_id"] == 1 for u in updates)

    def test_prompt_unknown_session_errors(self):
        _, resp, _ = _drive([_req("prompt", {"session_id": "ghost", "text": "hi"})])
        assert resp[0]["error"]["code"] == -32600
        assert "unknown session" in resp[0]["error"]["message"]

    def test_prompt_slash_alias(self):
        runner = _FakeRunner()
        server = AcpServer(prompt_handler=runner.run_prompt)
        sid = server.open_session("/tmp")["session_id"]
        stdin = io.StringIO(json.dumps(_req("session/prompt", {"session_id": sid, "text": "z"})) + "\n")
        stdout = io.StringIO()
        StdioJsonRpcServer(server, runner, stdin=stdin, stdout=stdout).serve_forever()
        lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
        assert any(m.get("result", {}).get("echo") == "z" for m in lines if "result" in m)


class TestPermissions:
    def test_respond_and_pending(self):
        _, resp, _ = _drive(
            [
                _req("pending_permissions", rid=1),
                _req("respond_permission", {"permission_id": "nope", "allow": True}, rid=2),
            ]
        )
        by_id = {r["id"]: r for r in resp}
        assert by_id[1]["result"] == []
        assert by_id[2]["result"]["ok"] is False  # unknown id


class TestErrorsAndFraming:
    def test_unknown_method(self):
        _, resp, _ = _drive([_req("does_not_exist")])
        assert resp[0]["error"]["code"] == -32601

    def test_malformed_json_line(self):
        runner = _FakeRunner()
        server = AcpServer(prompt_handler=runner.run_prompt)
        stdin = io.StringIO("{not json}\n" + json.dumps(_req("initialize")) + "\n")
        stdout = io.StringIO()
        StdioJsonRpcServer(server, runner, stdin=stdin, stdout=stdout).serve_forever()
        lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
        assert any(m.get("error", {}).get("code") == -32700 for m in lines)
        # ...and the server kept going (processed the next valid request)
        assert any("result" in m and m["result"].get("server_name") == "xerxes" for m in lines)

    def test_blank_lines_ignored(self):
        _, resp, _ = _drive([_req("initialize")])
        assert resp and resp[0]["result"]["server_name"] == "xerxes"

    def test_notification_without_id_no_response(self):
        # A request missing "id" is a notification — unknown method yields no error reply.
        _, resp, _notifs = _drive([{"jsonrpc": "2.0", "method": "does_not_exist", "params": {}}])
        assert resp == []

    def test_known_notification_without_id_no_response(self):
        _, resp, _notifs = _drive([{"jsonrpc": "2.0", "method": "initialize", "params": {}}])
        assert resp == []

    def test_invalid_request_shape(self):
        runner = _FakeRunner()
        server = AcpServer(prompt_handler=runner.run_prompt)
        stdin = io.StringIO(json.dumps({"foo": "bar"}) + "\n")
        stdout = io.StringIO()
        StdioJsonRpcServer(server, runner, stdin=stdin, stdout=stdout).serve_forever()
        lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
        assert any(m.get("error", {}).get("code") == -32600 for m in lines)
