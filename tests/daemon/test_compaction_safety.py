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
"""Resume safety for the daemon session manager.

* ``compact_if_needed`` no longer trims model messages; runtime compaction is
  handled by the agent-backed provisioner before provider calls.
* ``_repair_tool_pairs`` back-fills synthetic results for tool calls left
  orphaned by a mid-turn crash so ``xerxes -r`` rehydrates a valid history.
"""

from __future__ import annotations

from xerxes.daemon.runtime import SessionManager
from xerxes.streaming.events import AgentState
from xerxes.streaming.messages import messages_to_anthropic


class _StubWorkspace:
    def __init__(self):
        self.notes: list[str] = []

    def append_daily_note(self, note: str) -> None:
        self.notes.append(note)


class _StubSession:
    def __init__(self):
        self.id = "sess123"
        self.key = "tui:default"
        self.state = AgentState()
        self.workspace = _StubWorkspace()


def _mgr(tmp_path) -> SessionManager:
    mgr = SessionManager.__new__(SessionManager)
    mgr.keep_messages = 4
    mgr._store_dir = tmp_path
    return mgr


def test_compact_if_needed_preserves_model_messages(tmp_path):
    mgr = _mgr(tmp_path)
    s = _StubSession()
    s.state.messages = [
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "name": "X", "input": {}}]},
        {"role": "tool", "tool_call_id": "a", "name": "X", "content": "r"},
        {"role": "assistant", "content": "mid"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "b", "name": "Y", "input": {}}]},
        {"role": "tool", "tool_call_id": "b", "name": "Y", "content": "r2"},
    ]
    before = list(s.state.messages)
    assert mgr.compact_if_needed(s) is False
    assert s.state.messages == before
    assert s.state.messages[0]["role"] == "user"
    messages_to_anthropic(s.state.messages)


def test_compact_if_needed_never_empties_messages(tmp_path):
    mgr = _mgr(tmp_path)
    s = _StubSession()
    s.state.messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": f"c{i}", "name": "X", "input": {}} for i in range(8)],
        },
        *[{"role": "tool", "tool_call_id": f"c{i}", "name": "X", "content": "r"} for i in range(8)],
    ]
    before = list(s.state.messages)
    assert mgr.compact_if_needed(s) is False
    assert s.state.messages == before  # window preserved, not emptied
    assert len(s.state.messages) >= 1


def test_compact_if_needed_does_not_archive_messages(tmp_path):
    mgr = _mgr(tmp_path)
    s = _StubSession()
    s.state.messages = [{"role": "user", "content": f"m{i}"} for i in range(10)]
    assert mgr.compact_if_needed(s) is False
    archive = tmp_path / "sess123.archive.jsonl"
    assert not archive.exists()


def test_repair_backfills_orphaned_tool_call():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "a", "name": "X", "input": {}}, {"id": "b", "name": "Y", "input": {}}],
        },
        {"role": "tool", "tool_call_id": "a", "name": "X", "content": "done"},
        # 'b' never got a result (crash before it finished)
    ]
    out = SessionManager._repair_tool_pairs(messages)
    tool_ids = [m.get("tool_call_id") for m in out if m.get("role") == "tool"]
    assert "a" in tool_ids and "b" in tool_ids
    b_msg = next(m for m in out if m.get("tool_call_id") == "b")
    assert b_msg["is_error"] is True
    # The repaired history must convert without an orphan tool_use.
    messages_to_anthropic(out)


def test_repair_is_noop_when_complete():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "name": "X", "input": {}}]},
        {"role": "tool", "tool_call_id": "a", "name": "X", "content": "done"},
        {"role": "assistant", "content": "bye"},
    ]
    assert SessionManager._repair_tool_pairs(messages) == messages
