# Copyright 2026 Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
"""Coverage for the async sub-agent orchestration primitives.

Tests the new ``SubAgentManager`` event-bus / ``wait_for`` plumbing and the
four orchestration tools (``AwaitAgents``, ``CheckAgentMessages``,
``PeekAgent``, ``ResetAgent``) end-to-end with a stubbed runner so we do not
require a real LLM.

The contract under test mirrors the user-facing spec:
- spawn returns immediately and the main agent can keep working;
- ``wait_for`` blocks on a condition variable so it wakes promptly when an
  agent transitions (no polling delay);
- the mailbox accumulates events with monotonically-increasing seq numbers
  and drain returns only the new events;
- ``AwaitAgents`` honours its ``wake_on`` mode and wakes early on user
  input via ``session.pending_steers``;
- ``ResetAgent`` cancels the original and respawns with the recorded spec.
"""

from __future__ import annotations

import json
import queue
import threading
import time

import pytest
from xerxes.agents.subagent_manager import SubAgentManager, SubAgentTask
from xerxes.tools.claude_tools import (
    AwaitAgents,
    CheckAgentMessages,
    PeekAgent,
    ResetAgent,
)


@pytest.fixture
def mgr(monkeypatch):
    """Return a fresh global manager so tests do not share state."""
    m = SubAgentManager(max_concurrent=4, max_depth=5)
    monkeypatch.setattr("xerxes.tools.claude_tools._agent_manager", m)
    return m


def _spawn_stub(mgr: SubAgentManager, *, name: str, runtime_seconds: float = 0.05) -> SubAgentTask:
    """Spawn a task whose runner sleeps briefly then returns a fixed string."""

    def _runner(prompt, config, system, depth, cancel_check):
        end = time.monotonic() + runtime_seconds
        while time.monotonic() < end:
            if cancel_check():
                return "[cancelled]"
            time.sleep(0.005)
        return f"done:{prompt}"

    mgr.set_runner(_runner)
    return mgr.spawn(
        prompt=f"task-{name}",
        config={},
        system_prompt="",
        name=name,
    )


def test_wait_for_returns_true_when_predicate_satisfied(mgr):
    task = _spawn_stub(mgr, name="quick", runtime_seconds=0.05)
    ok = mgr.wait_for(lambda: task.status == "completed", timeout=2.0)
    assert ok is True
    assert task.status == "completed"


def test_wait_for_times_out_when_predicate_never_satisfied(mgr):
    start = time.monotonic()
    ok = mgr.wait_for(lambda: False, timeout=0.3)
    elapsed = time.monotonic() - start
    assert ok is False
    assert 0.25 < elapsed < 1.0


def test_wait_for_returns_false_on_extra_wake(mgr):
    fired = threading.Event()

    def _wake():
        return fired.is_set()

    def _trip():
        time.sleep(0.05)
        fired.set()

    threading.Thread(target=_trip, daemon=True).start()
    start = time.monotonic()
    ok = mgr.wait_for(lambda: False, timeout=5.0, extra_wake=_wake)
    elapsed = time.monotonic() - start
    assert ok is False
    assert elapsed < 2.0, f"extra_wake should fire well before timeout, took {elapsed}s"


def test_mailbox_records_spawn_and_done_events(mgr):
    task = _spawn_stub(mgr, name="watched", runtime_seconds=0.05)
    mgr.wait_for(lambda: task.status == "completed", timeout=2.0)
    events = mgr.peek_mailbox()
    types = [e["type"] for e in events]
    assert "spawn" in types
    assert "done" in types
    spawn_event = next(e for e in events if e["type"] == "spawn")
    assert spawn_event["agent"] == "watched"
    assert spawn_event["task_id"] == task.id


def test_drain_mailbox_clears_buffer(mgr):
    _spawn_stub(mgr, name="drain-target", runtime_seconds=0.05)
    mgr.wait_for(lambda: not mgr._mailbox or all(False for _ in []), timeout=0.1)
    # Give the runner time to post events.
    time.sleep(0.15)
    first = mgr.drain_mailbox()
    second = mgr.drain_mailbox()
    assert first, "first drain should return at least the spawn event"
    assert second == [], "second drain must be empty — events already consumed"


def test_drain_mailbox_since_seq_filters(mgr):
    _spawn_stub(mgr, name="seq-target", runtime_seconds=0.05)
    time.sleep(0.15)
    all_events = mgr.peek_mailbox()
    assert len(all_events) >= 2
    mid = all_events[0]["seq"]
    later = mgr.peek_mailbox(since_seq=mid)
    assert all(e["seq"] > mid for e in later)


def test_await_agents_wakes_when_agent_completes(mgr):
    task = _spawn_stub(mgr, name="await1", runtime_seconds=0.05)
    out = AwaitAgents.static_call(
        agent_ids=[task.id],
        wake_on="any",
        timeout_seconds=5.0,
    )
    payload = json.loads(out)
    assert payload["wake_reason"] == "agents_done"
    assert payload["agents"][0]["status"] == "completed"
    assert payload["elapsed_seconds"] < 1.0


def test_await_agents_times_out_when_no_agents(mgr):
    out = AwaitAgents.static_call(
        agent_ids=[],
        wake_on="any",
        timeout_seconds=0.2,
    )
    payload = json.loads(out)
    assert payload["wake_reason"] == "timeout"


def test_await_agents_wake_on_all_waits_for_everyone(mgr):
    fast = _spawn_stub(mgr, name="fast", runtime_seconds=0.05)
    # Re-spawn with a longer runner so the second task lags.
    def _slow(prompt, config, system, depth, cancel_check):
        end = time.monotonic() + 0.4
        while time.monotonic() < end:
            if cancel_check():
                return ""
            time.sleep(0.01)
        return "slow-done"

    mgr.set_runner(_slow)
    slow = mgr.spawn(prompt="slow", config={}, system_prompt="", name="slow")

    out = AwaitAgents.static_call(
        agent_ids=[fast.id, slow.id],
        wake_on="all",
        timeout_seconds=5.0,
    )
    payload = json.loads(out)
    assert payload["wake_reason"] == "agents_done"
    statuses = {a["name"]: a["status"] for a in payload["agents"]}
    assert statuses["fast"] == "completed"
    assert statuses["slow"] == "completed"


def test_await_agents_wakes_on_user_input(mgr):
    """A pending steer in the session must shortcut the sleep."""

    # Bind a fake session with a pending_steers queue so AwaitAgents sees user input.
    class _FakeSession:
        def __init__(self):
            self.pending_steers: queue.Queue[str] = queue.Queue()
            self.cancel_requested = False

    from xerxes.runtime.session_context import set_active_session

    session = _FakeSession()
    set_active_session(session)
    try:
        def _post_steer():
            time.sleep(0.05)
            session.pending_steers.put("hey")

        threading.Thread(target=_post_steer, daemon=True).start()
        # No agents — only the user_input path can wake us.
        out = AwaitAgents.static_call(agent_ids=[], wake_on="none", timeout_seconds=5.0)
        payload = json.loads(out)
        assert payload["wake_reason"] == "user_input"
        assert payload["elapsed_seconds"] < 2.0
    finally:
        set_active_session(None)


def test_await_agents_wakes_on_cancel(mgr):
    class _FakeSession:
        def __init__(self):
            self.pending_steers: queue.Queue[str] = queue.Queue()
            self.cancel_requested = False

    from xerxes.runtime.session_context import set_active_session

    session = _FakeSession()
    set_active_session(session)
    try:
        def _trip():
            time.sleep(0.05)
            session.cancel_requested = True

        threading.Thread(target=_trip, daemon=True).start()
        out = AwaitAgents.static_call(agent_ids=[], wake_on="none", timeout_seconds=5.0)
        payload = json.loads(out)
        assert payload["wake_reason"] == "cancelled"
    finally:
        set_active_session(None)


def test_check_agent_messages_drains_and_advances_cursor(mgr):
    _spawn_stub(mgr, name="checker", runtime_seconds=0.05)
    time.sleep(0.15)
    first = json.loads(CheckAgentMessages.static_call())
    assert first["events"], "must return queued events"
    second = json.loads(CheckAgentMessages.static_call())
    assert second["events"] == [], "drain should consume events"


def test_check_agent_messages_peek_does_not_consume(mgr):
    _spawn_stub(mgr, name="peeker", runtime_seconds=0.05)
    time.sleep(0.15)
    a = json.loads(CheckAgentMessages.static_call(peek=True))
    b = json.loads(CheckAgentMessages.static_call(peek=True))
    assert a["events"] == b["events"]


def test_peek_agent_returns_snapshot(mgr):
    task = _spawn_stub(mgr, name="peeker2", runtime_seconds=0.05)
    mgr.wait_for(lambda: task.status == "completed", timeout=2.0)
    out = json.loads(PeekAgent.static_call(target="peeker2"))
    assert out["name"] == "peeker2"
    assert out["status"] == "completed"
    assert "recent_output" in out
    assert "current_tool" in out
    assert "tool_calls_count" in out


def test_peek_agent_unknown_returns_error(mgr):
    out = PeekAgent.static_call(target="does-not-exist")
    assert "Error" in out


def test_reset_agent_cancels_and_respawns(mgr):
    # Long-running task so we have time to reset it.
    def _slow(prompt, config, system, depth, cancel_check):
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            if cancel_check():
                return "cancelled"
            time.sleep(0.01)
        return f"finished:{prompt}"

    mgr.set_runner(_slow)
    original = mgr.spawn(prompt="initial-prompt", config={}, system_prompt="", name="reset-target")

    # Give it a beat so it's actually running.
    time.sleep(0.05)
    out = json.loads(ResetAgent.static_call(target="reset-target", new_prompt="updated-prompt"))
    assert out["reset_target"] == "reset-target"
    new_task = out["new_task"]
    assert new_task["id"] != original.id, "reset must produce a new task id"
    assert new_task["name"] == "reset-target"
    assert new_task["prompt"].startswith("updated-prompt")
    # Original should now be cancelled.
    assert original.status in ("cancelled", "completed", "failed")


def test_reset_agent_unknown_returns_error(mgr):
    out = ResetAgent.static_call(target="missing")
    assert "Error" in out


def test_reset_agent_reuses_original_prompt_when_none_given(mgr):
    def _runner(prompt, config, system, depth, cancel_check):
        time.sleep(0.05)
        return prompt

    mgr.set_runner(_runner)
    original = mgr.spawn(prompt="keep-this-prompt", config={}, system_prompt="", name="reuse")
    mgr.wait_for(lambda: original.status == "completed", timeout=2.0)

    out = json.loads(ResetAgent.static_call(target="reuse"))
    assert "keep-this-prompt" in out["new_task"]["prompt"]
