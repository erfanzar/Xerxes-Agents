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
"""Tests for xerxes.mcp.server."""

from __future__ import annotations

from xerxes.mcp.server import MCP_TOOLS, DaemonBridge, XerxesMcpServer
from xerxes.session.models import SessionRecord, TurnRecord


class FakeSessions:
    def __init__(self, records):
        self._records = {r.session_id: r for r in records}

    def list_sessions(self, workspace_id=None):
        if workspace_id is None:
            return list(self._records.keys())
        return [sid for sid, r in self._records.items() if r.workspace_id == workspace_id]

    def get_session(self, session_id):
        return self._records.get(session_id)


def _make_server(records=None, **bridge_kwargs):
    rec = (
        records
        if records is not None
        else [
            SessionRecord(session_id="s1", workspace_id="w1"),
            SessionRecord(session_id="s2", workspace_id="w2"),
        ]
    )
    return XerxesMcpServer(sessions=FakeSessions(rec), bridge=DaemonBridge(**bridge_kwargs))


class TestToolTable:
    def test_ten_tools_exposed(self) -> None:
        assert len(MCP_TOOLS) == 10

    def test_required_tool_names(self) -> None:
        names = {t["name"] for t in MCP_TOOLS}
        assert names == {
            "conversations_list",
            "conversation_get",
            "messages_read",
            "attachments_fetch",
            "events_poll",
            "events_wait",
            "messages_send",
            "permissions_list_open",
            "permissions_respond",
            "channels_list",
        }

    def test_input_schema_present(self) -> None:
        for t in MCP_TOOLS:
            assert "input_schema" in t


class TestConversationsList:
    def test_returns_all_sessions(self):
        s = _make_server()
        out = s.conversations_list()
        ids = [r["session_id"] for r in out]
        assert ids == ["s1", "s2"]

    def test_scoped_by_workspace(self):
        s = _make_server()
        out = s.conversations_list(workspace_id="w1")
        assert [r["session_id"] for r in out] == ["s1"]


class TestConversationGet:
    def test_returns_dict(self):
        s = _make_server()
        out = s.conversation_get("s1")
        assert out is not None
        assert out["session_id"] == "s1"

    def test_missing_returns_none(self):
        s = _make_server()
        assert s.conversation_get("nope") is None


class TestMessagesRead:
    def test_returns_turns(self):
        rec = SessionRecord(
            session_id="s1",
            turns=[TurnRecord(turn_id="t1", prompt="hi"), TurnRecord(turn_id="t2", prompt="more")],
        )
        s = _make_server([rec])
        out = s.messages_read("s1")
        assert len(out) == 2
        assert out[0]["turn_id"] == "t1"

    def test_limit_takes_recent(self):
        rec = SessionRecord(
            session_id="s1",
            turns=[TurnRecord(turn_id=f"t{i}", prompt=str(i)) for i in range(5)],
        )
        s = _make_server([rec])
        out = s.messages_read("s1", limit=2)
        assert len(out) == 2
        assert out[0]["turn_id"] == "t3"

    def test_missing_session_returns_empty(self):
        s = _make_server()
        assert s.messages_read("nope") == []


class TestEventsPoll:
    def test_no_bridge_returns_empty(self):
        s = _make_server()
        assert s.events_poll("s1") == []

    def test_bridge_routes(self):
        calls = []

        def poll(sid, since):
            calls.append((sid, since))
            return [{"type": "Text", "text": "hi"}]

        s = _make_server(events_poll=poll)
        out = s.events_poll("s1", since_ts="abc")
        assert calls == [("s1", "abc")]
        assert out[0]["text"] == "hi"


class TestMessagesSend:
    def test_no_bridge_returns_error(self):
        s = _make_server()
        out = s.messages_send("s1", "hi")
        assert "error" in out

    def test_routes_to_bridge(self):
        captured = {}

        def send(sid, text, files=None):
            captured.update(dict(sid=sid, text=text, files=files))
            return {"ok": True}

        s = _make_server(send_message=send)
        out = s.messages_send("s1", "hi", files=["a.png"])
        assert out == {"ok": True}
        assert captured == {"sid": "s1", "text": "hi", "files": ["a.png"]}


class TestPermissions:
    def test_list_open_no_bridge_returns_empty(self):
        s = _make_server()
        assert s.permissions_list_open() == []

    def test_list_open_routes(self):
        s = _make_server(list_pending_permissions=lambda: [{"id": "p1"}, {"id": "p2"}])
        out = s.permissions_list_open()
        assert [p["id"] for p in out] == ["p1", "p2"]

    def test_respond_no_bridge_returns_error(self):
        s = _make_server()
        out = s.permissions_respond("p1", True)
        assert "error" in out

    def test_respond_routes(self):
        captured = {}

        def respond(pid, allow):
            captured.update(dict(pid=pid, allow=allow))
            return {"ok": True}

        s = _make_server(respond_permission=respond)
        out = s.permissions_respond("p1", False)
        assert out == {"ok": True}
        assert captured == {"pid": "p1", "allow": False}


class TestChannelsList:
    def test_no_bridge(self):
        s = _make_server()
        assert s.channels_list() == []

    def test_routes(self):
        s = _make_server(list_channels=lambda: [{"platform": "telegram", "id": "c1"}])
        out = s.channels_list()
        assert out[0]["platform"] == "telegram"


class TestRedaction:
    def test_api_key_redacted(self):
        records = [SessionRecord(session_id="s1", metadata={"api_key": "sk-secret"})]
        s = _make_server(records)
        out = s.conversation_get("s1")
        assert out["metadata"]["api_key"] == "[redacted]"
