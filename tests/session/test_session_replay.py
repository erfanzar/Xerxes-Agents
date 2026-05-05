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
"""Tests for session replay and timeline inspection."""

from __future__ import annotations

from xerxes.session.models import (
    AgentTransitionRecord,
    SessionRecord,
    ToolCallRecord,
    TurnRecord,
)
from xerxes.session.replay import ReplayView, SessionReplay
from xerxes.session.store import FileSessionStore, SessionManager


def _tc(call_id: str, name: str = "tool", **kw) -> ToolCallRecord:
    defaults = {
        "call_id": call_id,
        "tool_name": name,
        "arguments": {},
        "status": "success",
    }
    defaults.update(kw)
    return ToolCallRecord(**defaults)


def _turn(turn_id: str, agent_id: str = "a1", **kw) -> TurnRecord:
    defaults = {
        "turn_id": turn_id,
        "agent_id": agent_id,
        "prompt": f"prompt_{turn_id}",
        "response_content": f"response_{turn_id}",
        "tool_calls": [],
        "started_at": f"2025-01-01T00:00:0{turn_id[-1]}+00:00",
        "ended_at": f"2025-01-01T00:01:0{turn_id[-1]}+00:00",
        "status": "success",
    }
    defaults.update(kw)
    return TurnRecord(**defaults)


def _transition(from_a: str | None, to_a: str, ts: str = "") -> AgentTransitionRecord:
    return AgentTransitionRecord(
        from_agent=from_a,
        to_agent=to_a,
        reason="test",
        turn_id="t1",
        timestamp=ts or "2025-01-01T00:00:05+00:00",
    )


def _session(**kw) -> SessionRecord:
    defaults = {
        "session_id": "sess_replay",
        "workspace_id": "ws",
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:02:00+00:00",
        "agent_id": "a1",
        "turns": [
            _turn("t1", "a1", tool_calls=[_tc("tc1", "search"), _tc("tc2", "calc")]),
            _turn("t2", "a2", tool_calls=[_tc("tc3", "fetch")]),
            _turn("t3", "a1", tool_calls=[]),
        ],
        "agent_transitions": [
            _transition("a1", "a2", "2025-01-01T00:00:05+00:00"),
            _transition("a2", "a1", "2025-01-01T00:01:05+00:00"),
        ],
        "metadata": {},
    }
    defaults.update(kw)
    return SessionRecord(**defaults)


class TestReplayViewConstruction:
    def test_load_from_session(self):
        session = _session()
        view = SessionReplay.load(session)
        assert view.session is session
        assert len(view.turns) == 3

    def test_custom_turns(self):
        session = _session()
        view = ReplayView(session=session, turns=[session.turns[0]])
        assert len(view.turns) == 1


class TestGetTurn:
    def test_by_index(self):
        view = SessionReplay.load(_session())
        assert view.get_turn(0) is not None
        assert view.get_turn(0).turn_id == "t1"
        assert view.get_turn(2).turn_id == "t3"

    def test_by_id(self):
        view = SessionReplay.load(_session())
        assert view.get_turn("t2") is not None
        assert view.get_turn("t2").agent_id == "a2"

    def test_index_out_of_range(self):
        view = SessionReplay.load(_session())
        assert view.get_turn(99) is None
        assert view.get_turn(-1) is None

    def test_id_not_found(self):
        view = SessionReplay.load(_session())
        assert view.get_turn("nonexistent") is None


class TestGetToolCalls:
    def test_aggregates_across_turns(self):
        view = SessionReplay.load(_session())
        calls = view.get_tool_calls()
        assert len(calls) == 3
        names = [c.tool_name for c in calls]
        assert names == ["search", "calc", "fetch"]

    def test_empty_when_no_calls(self):
        session = _session(turns=[_turn("t1", tool_calls=[])])
        view = SessionReplay.load(session)
        assert view.get_tool_calls() == []


class TestGetAgentTransitions:
    def test_returns_all(self):
        view = SessionReplay.load(_session())
        transitions = view.get_agent_transitions()
        assert len(transitions) == 2
        assert transitions[0].to_agent == "a2"
        assert transitions[1].to_agent == "a1"


class TestGetTimeline:
    def test_chronological_order(self):
        view = SessionReplay.load(_session())
        timeline = view.get_timeline()
        timestamps = [e.timestamp for e in timeline]
        assert timestamps == sorted(timestamps)

    def test_event_types_present(self):
        view = SessionReplay.load(_session())
        timeline = view.get_timeline()
        types = {e.event_type for e in timeline}
        assert "turn_start" in types
        assert "turn_end" in types
        assert "tool_call" in types
        assert "agent_transition" in types

    def test_counts(self):
        view = SessionReplay.load(_session())
        timeline = view.get_timeline()
        starts = [e for e in timeline if e.event_type == "turn_start"]
        ends = [e for e in timeline if e.event_type == "turn_end"]
        tool_calls = [e for e in timeline if e.event_type == "tool_call"]
        transitions = [e for e in timeline if e.event_type == "agent_transition"]
        assert len(starts) == 3
        assert len(ends) == 3
        assert len(tool_calls) == 3
        assert len(transitions) == 2


class TestFilterByAgent:
    def test_returns_filtered_view(self):
        view = SessionReplay.load(_session())
        a1_view = view.filter_by_agent("a1")
        assert len(a1_view.turns) == 2
        for t in a1_view.turns:
            assert t.agent_id == "a1"

    def test_filtered_tool_calls(self):
        view = SessionReplay.load(_session())
        a2_view = view.filter_by_agent("a2")
        calls = a2_view.get_tool_calls()
        assert len(calls) == 1
        assert calls[0].tool_name == "fetch"

    def test_filter_unknown_agent(self):
        view = SessionReplay.load(_session())
        empty = view.filter_by_agent("nonexistent")
        assert len(empty.turns) == 0

    def test_session_reference_preserved(self):
        session = _session()
        view = SessionReplay.load(session)
        filtered = view.filter_by_agent("a1")

        assert filtered.session is session


class TestToMarkdown:
    def test_contains_session_id(self):
        md = SessionReplay.load(_session()).to_markdown()
        assert "sess_replay" in md

    def test_contains_turn_info(self):
        md = SessionReplay.load(_session()).to_markdown()
        assert "Turn 1:" in md or "t1" in md
        assert "Turn 2:" in md or "t2" in md

    def test_contains_tool_call_names(self):
        md = SessionReplay.load(_session()).to_markdown()
        assert "search" in md
        assert "calc" in md
        assert "fetch" in md

    def test_contains_agent_transition(self):
        md = SessionReplay.load(_session()).to_markdown()
        assert "a1" in md
        assert "a2" in md

    def test_contains_header(self):
        md = SessionReplay.load(_session()).to_markdown()
        assert md.startswith("# Session")

    def test_markdown_is_string(self):
        md = SessionReplay.load(_session()).to_markdown()
        assert isinstance(md, str)
        assert len(md) > 100


class TestRoundTrip:
    def test_full_round_trip(self, tmp_path):
        store = FileSessionStore(tmp_path)
        mgr = SessionManager(store)

        session = mgr.start_session(workspace_id="ws", agent_id="a1")
        sid = session.session_id

        t1 = _turn(
            "t1",
            "a1",
            tool_calls=[_tc("tc1", "search", arguments={"q": "test"})],
        )
        mgr.record_turn(sid, t1)

        mgr.record_agent_transition(sid, _transition("a1", "a2", "2025-01-01T00:00:05+00:00"))

        t2 = _turn("t2", "a2", tool_calls=[_tc("tc2", "calc")])
        mgr.record_turn(sid, t2)

        mgr.end_session(sid)

        store2 = FileSessionStore(tmp_path)
        loaded = store2.load_session(sid)
        assert loaded is not None

        view = SessionReplay.load(loaded)
        assert len(view.turns) == 2
        assert len(view.get_tool_calls()) == 2
        assert len(view.get_agent_transitions()) == 1

        timeline = view.get_timeline()
        timestamps = [e.timestamp for e in timeline]
        assert timestamps == sorted(timestamps)

        a1_view = view.filter_by_agent("a1")
        assert len(a1_view.turns) == 1
        assert a1_view.turns[0].turn_id == "t1"

        md = view.to_markdown()
        assert sid in md
        assert "search" in md
        assert "calc" in md
