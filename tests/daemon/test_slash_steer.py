# Copyright 2026 Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
"""Regression coverage for ``/steer`` and ``/btw`` actually reaching the loop.

The bulk smoke test only checks that *some* "Steer" string comes back to the
client, which the previous implementation satisfied while doing nothing. These
tests verify the real contract: pending steers land on the session's queue
when a turn is active, and on ``state.messages`` otherwise.
"""

from __future__ import annotations

import asyncio
import queue
from typing import Any

import pytest
from xerxes.daemon.config import DaemonConfig
from xerxes.daemon.runtime import DaemonSession, RuntimeManager
from xerxes.daemon.server import DaemonServer


class _FakeState:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((event_type, payload))


def _make_session(active_turn: str = "") -> DaemonSession:
    workspace = type("W", (), {"path": "/tmp/x"})()
    sess = DaemonSession.__new__(DaemonSession)
    sess.id = "sess1"
    sess.key = "tui:default"
    sess.agent_id = "default"
    sess.workspace = workspace
    sess.state = _FakeState()  # type: ignore[assignment]
    sess.lock = asyncio.Lock()
    sess.cancel_requested = False
    sess.active_turn_id = active_turn
    sess.pending_steers = queue.Queue()
    return sess


@pytest.fixture
def daemon(tmp_path):
    server = DaemonServer.__new__(DaemonServer)
    server.config = DaemonConfig(project_dir=str(tmp_path))
    server.runtime = RuntimeManager(server.config)
    server.runtime.runtime_config = {"model": "fake-model"}
    server._current_session_key = "tui:default"
    return server


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_steer_with_active_turn_queues_to_session(daemon):
    sess = _make_session(active_turn="turn-abc")
    daemon.sessions = type("S", (), {"get": lambda self_, key: sess})()

    rec = _Recorder()
    _run(daemon._slash_steer("course-correct now", rec))

    drained = sess.drain_steers()
    assert drained == ["course-correct now"]
    assert sess.state.messages == []
    # Wire event emitted for the TUI to render.
    assert any(et == "steer_input" for (et, _) in rec.events)


def test_steer_without_active_turn_lands_on_messages(daemon):
    sess = _make_session(active_turn="")
    daemon.sessions = type("S", (), {"get": lambda self_, key: sess})()

    rec = _Recorder()
    _run(daemon._slash_steer("hey think harder", rec))

    assert sess.drain_steers() == []
    assert len(sess.state.messages) == 1
    msg = sess.state.messages[0]
    assert msg["role"] == "user"
    assert "hey think harder" in msg["content"]


def test_steer_with_no_session_is_a_noop(daemon):
    daemon.sessions = type("S", (), {"get": lambda self_, key: None})()

    rec = _Recorder()
    _run(daemon._slash_steer("oops", rec))

    bodies = [
        p.get("body", "")
        for (et, p) in rec.events
        if et == "notification" and p.get("category") == "slash"
    ]
    assert any("no active session" in b.lower() for b in bodies)


def test_steer_empty_args_shows_usage(daemon):
    sess = _make_session(active_turn="turn-abc")
    daemon.sessions = type("S", (), {"get": lambda self_, key: sess})()

    rec = _Recorder()
    _run(daemon._slash_steer("   ", rec))

    assert sess.drain_steers() == []
    assert sess.state.messages == []


def test_drain_steers_returns_in_arrival_order():
    sess = _make_session(active_turn="x")
    sess.pending_steers.put("first")
    sess.pending_steers.put("second")
    sess.pending_steers.put("third")
    assert sess.drain_steers() == ["first", "second", "third"]
    assert sess.drain_steers() == []
