# Copyright 2026 Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
"""Verify ``compact_if_needed`` trims thinking_content + tool_executions.

``messages`` was the only buffer being trimmed; ``thinking_content`` and
``tool_executions`` accumulated forever and could account for the bulk of
session-state memory after hundreds of turns. They now ride along on the
same compaction path so a long session doesn't leak.
"""

from __future__ import annotations

from xerxes.daemon.runtime import SessionManager
from xerxes.streaming.events import AgentState


class _StubWorkspace:
    def __init__(self):
        self.notes: list[str] = []

    def append_daily_note(self, note: str) -> None:
        self.notes.append(note)


class _StubSession:
    def __init__(self):
        self.key = "tui:default"
        self.state = AgentState()
        self.workspace = _StubWorkspace()


def _make_mgr() -> SessionManager:
    mgr = SessionManager.__new__(SessionManager)
    mgr.keep_messages = 4
    return mgr


def test_thinking_content_trimmed_to_keep_limit():
    mgr = _make_mgr()
    session = _StubSession()
    session.state.thinking_content = [f"thought-{i}" for i in range(200)]
    assert mgr.compact_if_needed(session) is True
    assert len(session.state.thinking_content) == SessionManager._THINKING_KEEP
    # Tail preserved.
    assert session.state.thinking_content[-1] == "thought-199"


def test_tool_executions_trimmed_to_keep_limit():
    mgr = _make_mgr()
    session = _StubSession()
    session.state.tool_executions = [{"name": f"t{i}"} for i in range(500)]
    assert mgr.compact_if_needed(session) is True
    assert len(session.state.tool_executions) == SessionManager._TOOL_EXEC_KEEP
    assert session.state.tool_executions[-1]["name"] == "t499"


def test_returns_false_when_no_buffers_over_limit():
    mgr = _make_mgr()
    session = _StubSession()
    session.state.messages = [{"role": "user", "content": "hi"}]
    session.state.thinking_content = ["one"]
    session.state.tool_executions = []
    assert mgr.compact_if_needed(session) is False
