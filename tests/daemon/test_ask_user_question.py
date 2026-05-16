# Copyright 2026 Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
"""End-to-end coverage for the daemon's ``AskUserQuestionTool`` wiring.

The legacy bridge server installed an ``ask_user_question`` callback that
emitted a ``question_request`` wire event and parked the tool on a queue
until the TUI answered. The new daemon never did, so any agent that
called ``AskUserQuestionTool`` from inside a turn would hit the
non-interactive fallback (a plain string), the LLM would treat that as
text, and the user would never see an interactive panel. These tests
cover the new wiring: ``TurnRunner.ask_user_question`` emits the right
wire event, blocks on a queue, and is unblocked by the matching
``question_response`` RPC.
"""

from __future__ import annotations

import asyncio
import threading
import time

from xerxes.daemon.runtime import TurnRunner


def _make_runner() -> TurnRunner:
    """Build a ``TurnRunner`` shell without spinning the runtime/sessions."""
    runner = TurnRunner.__new__(TurnRunner)
    runner._pool = None  # not used in these tests
    runner._permission_lock = threading.Lock()
    runner._permission_waiters = {}
    runner._question_lock = threading.Lock()
    runner._question_waiters = {}
    runner._session_approvals = {}
    runner._subagent_buffer_lock = threading.Lock()
    runner._subagent_parent_tool = {}
    runner._subagent_tool_id_fifo = {}
    runner._subagent_text_buffers = {}
    runner._subagent_thinking_buffers = {}
    runner._current_tool_call_id = "tc-abc"
    runner._event_sink = None
    return runner


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_ask_user_question_emits_wire_event_and_blocks():
    """The tool callback must push a question_request and wait for an answer."""
    runner = _make_runner()
    emitted: list[tuple[str, dict]] = []

    def sink(event_type: str, payload: dict) -> None:
        emitted.append((event_type, payload))

    runner._event_sink = sink

    answer_holder: dict[str, str] = {}

    def _ask():
        answer_holder["value"] = runner.ask_user_question("What is the goal?")

    t = threading.Thread(target=_ask, daemon=True)
    t.start()

    # Wait for the wire event to land — the tool publishes synchronously
    # before parking on the queue.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not emitted:
        time.sleep(0.01)
    assert emitted, "ask_user_question must emit question_request before blocking"
    event_type, payload = emitted[0]
    assert event_type == "question_request"
    assert payload["questions"][0]["question"] == "What is the goal?"
    assert payload["questions"][0]["allow_free_form"] is True
    assert payload["tool_call_id"] == "tc-abc"
    request_id = payload["id"]

    # Caller is still blocked.
    assert "value" not in answer_holder
    assert t.is_alive()

    # Reply via the same path the daemon's question_response RPC uses.
    _run(runner.respond_question(request_id, {"q": "ship it"}))
    t.join(timeout=2.0)
    assert not t.is_alive(), "tool thread must unblock after respond_question"
    assert answer_holder["value"] == "ship it"


def test_ask_user_question_joins_multi_answer_dict():
    runner = _make_runner()
    runner._event_sink = lambda *_: None

    holder: dict[str, str] = {}

    def _ask():
        holder["value"] = runner.ask_user_question("multi?")

    t = threading.Thread(target=_ask, daemon=True)
    t.start()
    time.sleep(0.05)
    rid = next(iter(runner._question_waiters))
    _run(runner.respond_question(rid, {"a": "one", "b": "two"}))
    t.join(timeout=2.0)
    assert holder["value"] == "one\ntwo"


def test_ask_user_question_raises_when_no_sink():
    """A missing event sink is a daemon-bootstrap bug; never echo the question back."""
    import pytest

    runner = _make_runner()
    runner._event_sink = None
    with pytest.raises(RuntimeError, match="event sink"):
        runner.ask_user_question("anyone there?")


def test_respond_question_returns_false_for_unknown_id():
    runner = _make_runner()
    ok = _run(runner.respond_question("does-not-exist", {"q": "x"}))
    assert ok is False


def test_ask_user_question_tool_raises_when_callback_unwired():
    """If the daemon never installed a callback, the tool must NOT echo the question.

    Regression guard: the original tool fell back to a string that
    started with ``[AskUserQuestion]`` and contained the question text.
    The LLM treated that string as if the user had typed it, so a
    question like "Should we ship?" became a hallucinated user answer.
    Now the unwired path raises so callers see the bug immediately.
    """
    import pytest
    from xerxes.tools.claude_tools import AskUserQuestionTool, set_ask_user_question_callback

    set_ask_user_question_callback(None)
    try:
        with pytest.raises(RuntimeError, match="callback was never registered"):
            AskUserQuestionTool.static_call(question="Should we ship?")
    finally:
        set_ask_user_question_callback(None)


def test_ask_user_question_tool_routes_to_callback_when_wired():
    """Happy path — the wired callback's return value flows through unchanged."""
    from xerxes.tools.claude_tools import AskUserQuestionTool, set_ask_user_question_callback

    set_ask_user_question_callback(lambda q: f"echo:{q}")
    try:
        out = AskUserQuestionTool.static_call(question="hi")
        assert out == "echo:hi"
    finally:
        set_ask_user_question_callback(None)


def test_ask_user_question_unblocks_on_session_cancel():
    """A turn-level cancel must release the parked tool.

    The thread-local active-session pointer is per-thread, mirroring how
    the real daemon's ``TurnRunner._run_sync`` calls ``set_active_session``
    *on the worker thread* before the streaming loop invokes the tool. We
    set it inside the spawned thread for the same reason.
    """
    runner = _make_runner()
    runner._event_sink = lambda *_: None

    class _FakeSession:
        cancel_requested = False

    session = _FakeSession()

    from xerxes.runtime.session_context import set_active_session

    holder: dict[str, str] = {}

    def _ask():
        set_active_session(session)
        try:
            holder["value"] = runner.ask_user_question("blocking?")
        finally:
            set_active_session(None)

    t = threading.Thread(target=_ask, daemon=True)
    t.start()
    time.sleep(0.1)
    session.cancel_requested = True
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert holder["value"] == "[cancelled]"
