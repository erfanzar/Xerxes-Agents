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
"""Tests for session models, stores, and session manager."""

from __future__ import annotations

import json

import pytest
from xerxes.session.models import (
    AgentTransitionRecord,
    SessionRecord,
    ToolCallRecord,
    TurnRecord,
)
from xerxes.session.store import (
    FileSessionStore,
    InMemorySessionStore,
    SessionManager,
)


def _make_tool_call(**overrides) -> ToolCallRecord:
    defaults = {
        "call_id": "tc_1",
        "tool_name": "search",
        "arguments": {"query": "hello"},
        "result": "found it",
        "status": "success",
        "error": None,
        "duration_ms": 42.5,
        "sandbox_context": None,
    }
    defaults.update(overrides)
    return ToolCallRecord(**defaults)


def _make_turn(**overrides) -> TurnRecord:
    defaults = {
        "turn_id": "turn_1",
        "agent_id": "agent_a",
        "prompt": "What is 2+2?",
        "response_content": "4",
        "tool_calls": [_make_tool_call()],
        "started_at": "2025-01-01T00:00:00+00:00",
        "ended_at": "2025-01-01T00:00:01+00:00",
        "status": "success",
        "error": None,
        "audit_event_ids": ["evt_1"],
        "metadata": {"key": "value"},
    }
    defaults.update(overrides)
    return TurnRecord(**defaults)


def _make_transition(**overrides) -> AgentTransitionRecord:
    defaults = {
        "from_agent": "agent_a",
        "to_agent": "agent_b",
        "reason": "capability",
        "turn_id": "turn_1",
        "timestamp": "2025-01-01T00:00:00+00:00",
    }
    defaults.update(overrides)
    return AgentTransitionRecord(**defaults)


def _make_session(**overrides) -> SessionRecord:
    defaults = {
        "session_id": "sess_1",
        "workspace_id": "ws_1",
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:01+00:00",
        "agent_id": "agent_a",
        "turns": [_make_turn()],
        "agent_transitions": [_make_transition()],
        "metadata": {"env": "test"},
    }
    defaults.update(overrides)
    return SessionRecord(**defaults)


class TestToolCallRecordSerialization:
    def test_round_trip(self):
        tc = _make_tool_call()
        d = tc.to_dict()
        restored = ToolCallRecord.from_dict(d)
        assert restored.call_id == tc.call_id
        assert restored.tool_name == tc.tool_name
        assert restored.arguments == tc.arguments
        assert restored.result == tc.result
        assert restored.status == tc.status
        assert restored.error == tc.error
        assert restored.duration_ms == tc.duration_ms
        assert restored.sandbox_context == tc.sandbox_context

    def test_json_serializable(self):
        tc = _make_tool_call()
        # Should not raise
        json.dumps(tc.to_dict())

    def test_defaults(self):
        tc = ToolCallRecord.from_dict({"call_id": "x", "tool_name": "y", "arguments": {}})
        assert tc.status == "success"
        assert tc.error is None


class TestTurnRecordSerialization:
    def test_round_trip(self):
        turn = _make_turn()
        d = turn.to_dict()
        restored = TurnRecord.from_dict(d)
        assert restored.turn_id == turn.turn_id
        assert restored.agent_id == turn.agent_id
        assert restored.prompt == turn.prompt
        assert restored.response_content == turn.response_content
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].call_id == "tc_1"
        assert restored.started_at == turn.started_at
        assert restored.ended_at == turn.ended_at
        assert restored.status == turn.status
        assert restored.audit_event_ids == turn.audit_event_ids
        assert restored.metadata == turn.metadata

    def test_empty_tool_calls(self):
        turn = _make_turn(tool_calls=[])
        d = turn.to_dict()
        restored = TurnRecord.from_dict(d)
        assert restored.tool_calls == []

    def test_json_serializable(self):
        turn = _make_turn()
        json.dumps(turn.to_dict())


class TestAgentTransitionRecordSerialization:
    def test_round_trip(self):
        tr = _make_transition()
        d = tr.to_dict()
        restored = AgentTransitionRecord.from_dict(d)
        assert restored.from_agent == tr.from_agent
        assert restored.to_agent == tr.to_agent
        assert restored.reason == tr.reason
        assert restored.turn_id == tr.turn_id
        assert restored.timestamp == tr.timestamp

    def test_none_from_agent(self):
        tr = _make_transition(from_agent=None)
        d = tr.to_dict()
        restored = AgentTransitionRecord.from_dict(d)
        assert restored.from_agent is None


class TestSessionRecordSerialization:
    def test_round_trip(self):
        session = _make_session()
        d = session.to_dict()
        restored = SessionRecord.from_dict(d)
        assert restored.session_id == session.session_id
        assert restored.workspace_id == session.workspace_id
        assert restored.created_at == session.created_at
        assert restored.updated_at == session.updated_at
        assert restored.agent_id == session.agent_id
        assert len(restored.turns) == 1
        assert len(restored.agent_transitions) == 1
        assert restored.metadata == session.metadata

    def test_json_round_trip(self):
        session = _make_session()
        json_str = json.dumps(session.to_dict())
        data = json.loads(json_str)
        restored = SessionRecord.from_dict(data)
        assert restored.session_id == session.session_id

    def test_empty_session(self):
        session = SessionRecord(session_id="empty")
        d = session.to_dict()
        restored = SessionRecord.from_dict(d)
        assert restored.session_id == "empty"
        assert restored.turns == []
        assert restored.agent_transitions == []


class TestInMemorySessionStore:
    def test_save_and_load(self):
        store = InMemorySessionStore()
        session = _make_session()
        store.save_session(session)
        loaded = store.load_session("sess_1")
        assert loaded is not None
        assert loaded.session_id == "sess_1"

    def test_load_missing(self):
        store = InMemorySessionStore()
        assert store.load_session("nonexistent") is None

    def test_list_all(self):
        store = InMemorySessionStore()
        store.save_session(_make_session(session_id="s1", workspace_id="w1"))
        store.save_session(_make_session(session_id="s2", workspace_id="w2"))
        ids = store.list_sessions()
        assert set(ids) == {"s1", "s2"}

    def test_list_by_workspace(self):
        store = InMemorySessionStore()
        store.save_session(_make_session(session_id="s1", workspace_id="w1"))
        store.save_session(_make_session(session_id="s2", workspace_id="w2"))
        store.save_session(_make_session(session_id="s3", workspace_id="w1"))
        ids = store.list_sessions(workspace_id="w1")
        assert set(ids) == {"s1", "s3"}

    def test_delete(self):
        store = InMemorySessionStore()
        store.save_session(_make_session())
        assert store.delete_session("sess_1") is True
        assert store.load_session("sess_1") is None

    def test_delete_missing(self):
        store = InMemorySessionStore()
        assert store.delete_session("nonexistent") is False

    def test_overwrite(self):
        store = InMemorySessionStore()
        store.save_session(_make_session(metadata={"v": 1}))
        store.save_session(_make_session(metadata={"v": 2}))
        loaded = store.load_session("sess_1")
        assert loaded is not None
        assert loaded.metadata["v"] == 2


class TestFileSessionStore:
    def test_save_and_load(self, tmp_path):
        store = FileSessionStore(tmp_path)
        session = _make_session()
        store.save_session(session)
        loaded = store.load_session("sess_1")
        assert loaded is not None
        assert loaded.session_id == "sess_1"
        assert len(loaded.turns) == 1

    def test_load_missing(self, tmp_path):
        store = FileSessionStore(tmp_path)
        assert store.load_session("nope") is None

    def test_workspace_subdirectory(self, tmp_path):
        store = FileSessionStore(tmp_path)
        session = _make_session(workspace_id="ws_x")
        store.save_session(session)
        # File should exist under workspace subdirectory
        assert (tmp_path / "ws_x" / "sess_1.json").exists()
        loaded = store.load_session("sess_1")
        assert loaded is not None

    def test_flat_layout_when_no_workspace(self, tmp_path):
        store = FileSessionStore(tmp_path)
        session = _make_session(workspace_id=None)
        store.save_session(session)
        assert (tmp_path / "sess_1.json").exists()

    def test_list_all(self, tmp_path):
        store = FileSessionStore(tmp_path)
        store.save_session(_make_session(session_id="s1", workspace_id=None))
        store.save_session(_make_session(session_id="s2", workspace_id="ws_a"))
        ids = store.list_sessions()
        assert set(ids) == {"s1", "s2"}

    def test_list_by_workspace(self, tmp_path):
        store = FileSessionStore(tmp_path)
        store.save_session(_make_session(session_id="s1", workspace_id="ws_a"))
        store.save_session(_make_session(session_id="s2", workspace_id="ws_b"))
        ids = store.list_sessions(workspace_id="ws_a")
        assert ids == ["s1"]

    def test_delete(self, tmp_path):
        store = FileSessionStore(tmp_path)
        store.save_session(_make_session())
        assert store.delete_session("sess_1") is True
        assert store.load_session("sess_1") is None

    def test_delete_missing(self, tmp_path):
        store = FileSessionStore(tmp_path)
        assert store.delete_session("nonexistent") is False

    def test_json_content_valid(self, tmp_path):
        store = FileSessionStore(tmp_path)
        session = _make_session(workspace_id=None)
        store.save_session(session)
        path = tmp_path / "sess_1.json"
        data = json.loads(path.read_text())
        assert data["session_id"] == "sess_1"


class TestSessionManager:
    def test_start_session(self):
        mgr = SessionManager(InMemorySessionStore())
        session = mgr.start_session(workspace_id="w1", agent_id="a1")
        assert session.workspace_id == "w1"
        assert session.agent_id == "a1"
        assert session.created_at != ""
        assert session.session_id != ""

    def test_start_session_explicit_id(self):
        mgr = SessionManager(InMemorySessionStore())
        session = mgr.start_session(session_id="my_id")
        assert session.session_id == "my_id"

    def test_record_turn(self):
        mgr = SessionManager(InMemorySessionStore())
        session = mgr.start_session()
        turn = _make_turn()
        mgr.record_turn(session.session_id, turn)
        loaded = mgr.get_session(session.session_id)
        assert loaded is not None
        assert len(loaded.turns) == 1
        assert loaded.turns[0].turn_id == "turn_1"

    def test_record_turn_missing_session(self):
        mgr = SessionManager(InMemorySessionStore())
        with pytest.raises(ValueError, match="Session not found"):
            mgr.record_turn("nonexistent", _make_turn())

    def test_record_agent_transition(self):
        mgr = SessionManager(InMemorySessionStore())
        session = mgr.start_session()
        tr = _make_transition()
        mgr.record_agent_transition(session.session_id, tr)
        loaded = mgr.get_session(session.session_id)
        assert loaded is not None
        assert len(loaded.agent_transitions) == 1

    def test_record_agent_transition_missing_session(self):
        mgr = SessionManager(InMemorySessionStore())
        with pytest.raises(ValueError, match="Session not found"):
            mgr.record_agent_transition("nonexistent", _make_transition())

    def test_end_session(self):
        mgr = SessionManager(InMemorySessionStore())
        session = mgr.start_session()
        mgr.end_session(session.session_id)
        loaded = mgr.get_session(session.session_id)
        assert loaded is not None
        assert loaded.metadata.get("ended") is True

    def test_end_session_missing(self):
        mgr = SessionManager(InMemorySessionStore())
        with pytest.raises(ValueError, match="Session not found"):
            mgr.end_session("nonexistent")

    def test_list_sessions(self):
        mgr = SessionManager(InMemorySessionStore())
        mgr.start_session(workspace_id="w1", session_id="s1")
        mgr.start_session(workspace_id="w2", session_id="s2")
        assert set(mgr.list_sessions()) == {"s1", "s2"}
        assert mgr.list_sessions(workspace_id="w1") == ["s1"]

    def test_get_session_missing(self):
        mgr = SessionManager(InMemorySessionStore())
        assert mgr.get_session("nonexistent") is None

    def test_full_lifecycle(self, tmp_path):
        """End-to-end: start, record turns, transition, end, reload."""
        store = FileSessionStore(tmp_path)
        mgr = SessionManager(store)

        session = mgr.start_session(workspace_id="ws", agent_id="a1")
        sid = session.session_id

        mgr.record_turn(sid, _make_turn(turn_id="t1"))
        mgr.record_agent_transition(sid, _make_transition(from_agent="a1", to_agent="a2"))
        mgr.record_turn(sid, _make_turn(turn_id="t2", agent_id="a2"))
        mgr.end_session(sid)

        # Reload from a fresh store pointing at the same directory
        store2 = FileSessionStore(tmp_path)
        mgr2 = SessionManager(store2)
        loaded = mgr2.get_session(sid)
        assert loaded is not None
        assert len(loaded.turns) == 2
        assert len(loaded.agent_transitions) == 1
        assert loaded.metadata.get("ended") is True
