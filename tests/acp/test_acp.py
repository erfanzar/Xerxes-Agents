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
"""Tests for the ACP adapter."""

from __future__ import annotations

import json
import threading

from xerxes.acp import (
    REGISTRY_METADATA,
    AcpEventKind,
    AcpPermissionRequest,
    AcpServer,
    AcpSession,
    AcpSessionStore,
    ServerCapabilities,
    to_acp_event,
    write_registry_file,
)
from xerxes.acp.permissions import AcpPermissionBoard, route_permission
from xerxes.streaming.events import (
    PermissionRequest,
    SkillSuggestion,
    TextChunk,
    ThinkingChunk,
    ToolEnd,
    ToolStart,
    TurnDone,
)


class TestRegistry:
    def test_metadata_shape(self):
        assert REGISTRY_METADATA["name"] == "xerxes"
        assert REGISTRY_METADATA["distribution"]["type"] == "command"
        assert REGISTRY_METADATA["distribution"]["command"] == "xerxes-acp"

    def test_write_registry_file(self, tmp_path):
        out = write_registry_file(tmp_path)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["name"] == "xerxes"


class TestEventConversion:
    def test_text_chunk(self):
        ev = to_acp_event(TextChunk("hi"))
        assert ev.kind is AcpEventKind.TEXT_DELTA
        assert ev.payload["text"] == "hi"

    def test_thinking_chunk(self):
        ev = to_acp_event(ThinkingChunk("plan"))
        assert ev.kind is AcpEventKind.THINKING_DELTA

    def test_tool_start_end(self):
        s = to_acp_event(ToolStart("read", {"path": "a"}, tool_call_id="x"))
        assert s.kind is AcpEventKind.TOOL_CALL_START
        assert s.payload["name"] == "read"
        e = to_acp_event(ToolEnd("read", "ok", tool_call_id="x", duration_ms=12.5))
        assert e.kind is AcpEventKind.TOOL_CALL_END
        assert e.payload["duration_ms"] == 12.5

    def test_permission_request(self):
        ev = to_acp_event(PermissionRequest("rm_rf", "danger", {"path": "/"}))
        assert ev.kind is AcpEventKind.PERMISSION_REQUEST
        assert ev.payload["tool_name"] == "rm_rf"

    def test_turn_done_carries_cache_tokens(self):
        td = TurnDone(input_tokens=1, output_tokens=2, cache_read_tokens=100, cache_creation_tokens=50)
        ev = to_acp_event(td)
        assert ev.payload["cache_read_tokens"] == 100
        assert ev.payload["cache_creation_tokens"] == 50

    def test_skill_suggestion(self):
        ev = to_acp_event(SkillSuggestion("my-skill"))
        assert ev.kind is AcpEventKind.SKILL_SUGGESTION

    def test_unknown_event_handled(self):
        # Anything not in the dispatch table falls through.
        class _Other:
            pass

        ev = to_acp_event(_Other())  # type: ignore[arg-type]
        assert ev.kind is AcpEventKind.UNKNOWN


class TestSessionStore:
    def test_create_and_lookup(self):
        store = AcpSessionStore()
        s = store.create("/tmp", model="gpt-4o", title="T")
        assert isinstance(s, AcpSession)
        assert store.get(s.session_id) is s

    def test_set_model(self):
        store = AcpSessionStore()
        s = store.create("/tmp")
        assert store.set_model(s.session_id, "claude-opus-4-7") is True
        assert store.get(s.session_id).model_override == "claude-opus-4-7"

    def test_cancel_marks_session(self):
        store = AcpSessionStore()
        s = store.create("/tmp")
        store.cancel(s.session_id)
        assert store.get(s.session_id).cancelled is True

    def test_drop_removes(self):
        store = AcpSessionStore()
        s = store.create("/tmp")
        assert store.drop(s.session_id) is True
        assert store.get(s.session_id) is None


class TestPermissionBoard:
    def test_submit_and_resolve(self):
        board = AcpPermissionBoard()
        req = route_permission(session_id="s1", tool_name="t", description="d", inputs={})
        ev = board.submit(req)
        threading.Timer(0.01, lambda: board.resolve(req.id, True)).start()
        assert ev.wait(timeout=1.0) is True
        assert board.get(req.id).allowed is True

    def test_resolve_twice_fails(self):
        board = AcpPermissionBoard()
        req = route_permission(session_id="s1", tool_name="t", description="d", inputs={})
        board.submit(req)
        board.resolve(req.id, True)
        assert board.resolve(req.id, False) is False

    def test_snapshot_excludes_decided(self):
        board = AcpPermissionBoard()
        a = route_permission(session_id="s", tool_name="a", description="", inputs={})
        b = route_permission(session_id="s", tool_name="b", description="", inputs={})
        board.submit(a)
        board.submit(b)
        board.resolve(a.id, True)
        pending = board.snapshot_pending()
        assert [p.id for p in pending] == [b.id]


class TestAcpServer:
    def _server(self, **kw):
        kw.setdefault("prompt_handler", lambda **kwargs: {"echo": kwargs.get("text")})
        return AcpServer(**kw)

    def test_initialize(self):
        s = self._server()
        out = s.initialize()
        assert out["server_name"] == "xerxes"
        assert out["capabilities"]["protocol_version"] == "0.9"

    def test_open_and_list_sessions(self):
        s = self._server()
        opened = s.open_session("/tmp", model="claude-opus-4-7")
        sid = opened["session_id"]
        lst = s.list_sessions()
        assert any(x["session_id"] == sid for x in lst)

    def test_set_model_ok(self):
        s = self._server()
        sid = s.open_session("/tmp")["session_id"]
        out = s.set_model(sid, "gpt-4o")
        assert out["ok"] is True

    def test_cancel_and_close(self):
        s = self._server()
        sid = s.open_session("/tmp")["session_id"]
        assert s.cancel(sid)["ok"] is True
        assert s.close_session(sid)["ok"] is True

    def test_prompt_routes_to_handler(self):
        captured = {}

        def handler(*, session, text, **_):
            captured["text"] = text
            captured["sid"] = session.session_id
            return {"ok": True}

        s = self._server(prompt_handler=handler)
        sid = s.open_session("/tmp")["session_id"]
        out = s.prompt(sid, "hello")
        assert out == {"ok": True}
        assert captured == {"text": "hello", "sid": sid}

    def test_prompt_unknown_session_returns_error(self):
        s = self._server()
        out = s.prompt("nope", "hi")
        assert "error" in out

    def test_list_tools_passthrough(self):
        s = self._server(tool_list_provider=lambda: [{"name": "x"}])
        assert s.list_tools() == [{"name": "x"}]

    def test_request_and_respond_permission(self):
        s = self._server()
        out = s.request_permission(session_id="s1", tool_name="t", description="d", inputs={})
        pid = out["permission_id"]
        pending = s.pending_permissions()
        assert any(p["id"] == pid for p in pending)
        assert s.respond_permission(pid, True) == {"ok": True}
        # Now drained.
        assert s.pending_permissions() == []

    def test_stream_event_wraps_internal(self):
        s = self._server()
        out = s.stream_event(TextChunk("hi"))
        assert out["kind"] == "text_delta"
        assert out["text"] == "hi"


class TestServerCapabilities:
    def test_defaults(self):
        c = ServerCapabilities()
        d = c.to_dict()
        assert d["streaming"] is True
        assert d["tools"] is True
        assert d["permissions"] is True
        assert d["fork"] is True


class TestPermissionRequestDataclass:
    def test_construction_defaults(self):
        r = AcpPermissionRequest(id="x", session_id="s", tool_name="t", description="d", inputs={"a": 1})
        assert r.decided is False
        assert r.allowed is False
