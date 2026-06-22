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
"""Verify ``compact_if_needed`` trims thinking_content + tool_executions."""

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
