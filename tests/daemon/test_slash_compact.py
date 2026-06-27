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
"""Regression coverage for daemon ``/compact`` rewriting session state."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from xerxes.daemon import slash_commands
from xerxes.daemon.config import DaemonConfig
from xerxes.daemon.runtime import RuntimeManager
from xerxes.daemon.server import DaemonServer
from xerxes.streaming.events import AgentState

from tests.async_helpers import run_coro


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((event_type, payload))

    def bodies(self) -> list[str]:
        return [
            payload.get("body", "")
            for event_type, payload in self.events
            if event_type == "notification" and payload.get("category") == "slash"
        ]


class _Sessions:
    def __init__(self, session: Any) -> None:
        self.session = session
        self.saved = False

    def get(self, key: str) -> Any:
        return self.session

    def save(self, session: Any) -> None:
        assert session is self.session
        self.saved = True


def _run(coro):
    return run_coro(coro)


def test_slash_compact_rewrites_session_messages_and_saves(tmp_path, monkeypatch):
    server = DaemonServer.__new__(DaemonServer)
    server.config = DaemonConfig(project_dir=str(tmp_path))
    server.runtime = RuntimeManager(server.config)
    server.runtime.runtime_config = {
        "model": "gpt-4o",
        "max_context_tokens": 120,
        "compaction_target_tokens": 20,
    }
    server.runtime.tool_schemas = []
    server.runtime.system_prompt = ""
    server._current_session_key = "tui:default"
    server._connection_session_key = lambda _emit: "tui:default"

    async def emit_slash(emit, text: str) -> None:
        await emit("notification", {"category": "slash", "body": text})

    async def emit_status(emit) -> None:
        await emit("status_update", {"context_tokens": 1, "max_context": 120})

    async def submit_turn(_params, _emit):
        raise AssertionError("/compact must not submit a normal model turn")

    server._emit_slash = emit_slash
    server._emit_status = emit_status
    server._submit_turn = submit_turn

    state = AgentState(
        messages=[
            {"role": "user", "content": "old user " * 120},
            {"role": "assistant", "content": "old assistant " * 120},
            {"role": "user", "content": "current question"},
            {"role": "assistant", "content": "current answer"},
        ]
    )
    session = SimpleNamespace(
        id="sess1",
        state=state,
        runtime_config={"model": "gpt-4o", "max_context_tokens": 120, "compaction_target_tokens": 20},
    )
    sessions = _Sessions(session)
    server.sessions = sessions

    monkeypatch.setattr(
        slash_commands,
        "compaction_summary_agent_from_config",
        lambda _model, _config: lambda _messages, _previous: "agent-written summary",
    )

    rec = _Recorder()
    _run(server._slash_compact("", rec))

    assert sessions.saved is True
    assert len(session.state.messages) < 4
    assert "agent-written summary" in session.state.messages[0]["content"]
    assert any("Compacted 4 messages" in body for body in rec.bodies())
    assert any(event_type == "status_update" for event_type, _payload in rec.events)
