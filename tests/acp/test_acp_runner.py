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
"""Deterministic tests for AcpAgentRunner.run_prompt.

The streaming loop is monkeypatched to yield canned events, so we can verify
event→ACP conversion, the interactive permission board flow (allow / deny /
cancel), the no-board auto-deny, multi-turn AgentState persistence, and error
handling — all without a live LLM."""

from __future__ import annotations

import threading
import time
import types

import pytest
from xerxes.acp.permissions import AcpPermissionBoard
from xerxes.acp.runner import AcpAgentRunner
from xerxes.acp.session import AcpSession
from xerxes.streaming.events import PermissionRequest, TextChunk, ToolEnd, ToolStart, TurnDone


def _make_runner(default_permission_mode="accept-all") -> AcpAgentRunner:
    """Build a runner without bootstrapping a real RuntimeManager."""
    r = AcpAgentRunner.__new__(AcpAgentRunner)
    r._runtime = types.SimpleNamespace(
        system_prompt="",
        tool_executor=lambda name, inputs: "tool-result",
        tool_schemas=[],
        runtime_config={"model": "m", "base_url": "", "api_key": ""},
        model="m",
    )
    r._states = {}
    r._default_permission_mode = default_permission_mode
    r.permission_board = None
    r._questions = {}
    r._q_lock = threading.Lock()
    r._ctx = threading.local()
    return r


def _session(**kw) -> AcpSession:
    return AcpSession(session_id="s1", cwd="/tmp", **kw)


def _install_fake_loop(monkeypatch, events_factory):
    """Patch run_loop to a generator built by ``events_factory(req_holder)``."""
    import xerxes.acp.runner as runner_mod

    def fake_loop(*, user_message, state, config, system_prompt, tool_executor, tool_schemas, cancel_check, **_):
        state.messages.append({"role": "user", "content": user_message})
        yield from events_factory(state, config, cancel_check)
        state.messages.append({"role": "assistant", "content": "done"})

    monkeypatch.setattr(runner_mod, "run_loop", fake_loop)


class TestEventStreaming:
    def test_text_and_tool_events_convert_to_acp(self, monkeypatch):
        def events(state, config, cancel_check):
            yield TextChunk("hi ")
            yield ToolStart(name="grep", inputs={"q": "x"}, tool_call_id="t1")
            yield ToolEnd(name="grep", result="r", permitted=True, tool_call_id="t1", duration_ms=1.0)
            yield TurnDone(input_tokens=3, output_tokens=4, tool_calls_count=1, model="m")

        _install_fake_loop(monkeypatch, events)
        out = []
        r = _make_runner()
        summary = r.run_prompt(session=_session(), text="hello", emit=out.append)
        kinds = [e["kind"] for e in out]
        assert kinds == ["text_delta", "tool_call_start", "tool_call_end", "turn_end"]
        assert out[0]["text"] == "hi "
        assert out[1]["name"] == "grep" and out[1]["tool_call_id"] == "t1"
        assert summary["ok"] is True and summary["output_tokens"] == 4 and summary["cancelled"] is False

    def test_multi_turn_state_persists(self, monkeypatch):
        def events(state, config, cancel_check):
            yield TextChunk("ok")
            yield TurnDone(input_tokens=1, output_tokens=1, tool_calls_count=0, model="m")

        _install_fake_loop(monkeypatch, events)
        r = _make_runner()
        sess = _session()
        r.run_prompt(session=sess, text="first", emit=lambda e: None)
        r.run_prompt(session=sess, text="second", emit=lambda e: None)
        msgs = r._states[sess.session_id].messages
        # 2 user + 2 assistant messages accumulated across turns
        assert [m["content"] for m in msgs if m["role"] == "user"] == ["first", "second"]
        assert sum(1 for m in msgs if m["role"] == "assistant") == 2


class TestPermissionResolution:
    def _perm_events(self):
        captured = {}

        def events(state, config, cancel_check):
            req = PermissionRequest(tool_name="write_file", description="write", inputs={"path": "x"})
            yield req
            captured["granted"] = req.granted
            yield ToolEnd(
                name="write_file", result="ok" if req.granted else "denied", permitted=req.granted, tool_call_id="t1"
            )
            yield TurnDone(input_tokens=1, output_tokens=1, tool_calls_count=1, model="m")

        return events, captured

    def test_no_board_auto_denies(self, monkeypatch):
        events, captured = self._perm_events()
        _install_fake_loop(monkeypatch, events)
        r = _make_runner()
        r.permission_board = None
        out = []
        r.run_prompt(session=_session(), text="go", emit=out.append)
        assert captured["granted"] is False
        # the permission_request was still surfaced to the client
        assert any(e["kind"] == "permission_request" for e in out)

    def test_board_allow_grants(self, monkeypatch):
        events, captured = self._perm_events()
        _install_fake_loop(monkeypatch, events)
        r = _make_runner()
        board = r.permission_board = AcpPermissionBoard()

        def resolver():
            for _ in range(200):
                pend = board.snapshot_pending()
                if pend:
                    board.resolve(pend[0].id, True)
                    return
                time.sleep(0.01)

        t = threading.Thread(target=resolver)
        t.start()
        r.run_prompt(session=_session(), text="go", emit=lambda e: None)
        t.join(timeout=3)
        assert captured["granted"] is True

    def test_board_deny(self, monkeypatch):
        events, captured = self._perm_events()
        _install_fake_loop(monkeypatch, events)
        r = _make_runner()
        board = r.permission_board = AcpPermissionBoard()

        def resolver():
            for _ in range(200):
                pend = board.snapshot_pending()
                if pend:
                    board.resolve(pend[0].id, False)
                    return
                time.sleep(0.01)

        t = threading.Thread(target=resolver)
        t.start()
        r.run_prompt(session=_session(), text="go", emit=lambda e: None)
        t.join(timeout=3)
        assert captured["granted"] is False

    def test_cancel_during_permission_denies(self, monkeypatch):
        events, captured = self._perm_events()
        _install_fake_loop(monkeypatch, events)
        r = _make_runner()
        board = r.permission_board = AcpPermissionBoard()
        sess = _session()

        def canceller():
            for _ in range(200):
                if board.snapshot_pending():
                    sess.cancelled = True
                    return
                time.sleep(0.01)

        t = threading.Thread(target=canceller)
        t.start()
        summary = r.run_prompt(session=sess, text="go", emit=lambda e: None)
        t.join(timeout=3)
        assert captured["granted"] is False
        assert summary["cancelled"] is True


class TestErrorHandling:
    def test_loop_exception_returns_error_summary(self, monkeypatch):
        import xerxes.acp.runner as runner_mod

        def boom(**_):
            raise RuntimeError("kaboom")
            yield  # pragma: no cover

        monkeypatch.setattr(runner_mod, "run_loop", boom)
        r = _make_runner()
        summary = r.run_prompt(session=_session(), text="x", emit=lambda e: None)
        assert summary["ok"] is False and "kaboom" in summary["error"]


class TestAskUserQuestion:
    """The blocking AskUserQuestion callback routes through the ACP client."""

    def test_question_routes_and_resolves(self, monkeypatch):
        r = _make_runner()
        emitted: list = []
        captured: dict = {}

        def events(state, config, cancel_check):
            captured["answer"] = r.ask_user_question("Pick a color?")  # called during tool exec
            yield TurnDone(input_tokens=0, output_tokens=0, tool_calls_count=0, model="m")

        _install_fake_loop(monkeypatch, events)

        def responder():
            for _ in range(300):
                pend = r.pending_questions()
                if pend:
                    r.respond_question(pend[0]["input_id"], "blue")
                    return
                time.sleep(0.01)

        t = threading.Thread(target=responder)
        t.start()
        r.run_prompt(session=_session(), text="go", emit=emitted.append)
        t.join(timeout=3)
        assert captured["answer"] == "blue"
        reqs = [e for e in emitted if e.get("kind") == "input_request"]
        assert reqs and reqs[0]["question"] == "Pick a color?" and reqs[0]["session_id"] == "s1"

    def test_question_outside_turn_raises(self):
        r = _make_runner()
        with pytest.raises(RuntimeError):
            r.ask_user_question("no active turn")

    def test_cancel_during_question_returns_empty(self, monkeypatch):
        r = _make_runner()
        sess = _session()
        captured: dict = {}

        def events(state, config, cancel_check):
            captured["answer"] = r.ask_user_question("blocking?")
            yield TurnDone(input_tokens=0, output_tokens=0, tool_calls_count=0, model="m")

        _install_fake_loop(monkeypatch, events)

        def canceller():
            for _ in range(300):
                if r.pending_questions():
                    sess.cancelled = True
                    return
                time.sleep(0.01)

        t = threading.Thread(target=canceller)
        t.start()
        r.run_prompt(session=sess, text="go", emit=lambda e: None)
        t.join(timeout=3)
        assert captured["answer"] == ""

    def test_respond_unknown_question(self):
        r = _make_runner()
        assert r.respond_question("nope", "x")["ok"] is False


class TestModelOverride:
    def test_model_override_feeds_config(self, monkeypatch):
        seen = {}

        def events(state, config, cancel_check):
            seen["model"] = config.get("model")
            seen["perm"] = config.get("permission_mode")
            yield TurnDone(input_tokens=0, output_tokens=0, tool_calls_count=0, model=config.get("model", ""))

        _install_fake_loop(monkeypatch, events)
        r = _make_runner(default_permission_mode="accept-all")
        r.run_prompt(session=_session(model_override="custom-model"), text="x", emit=lambda e: None)
        assert seen["model"] == "custom-model"
        assert seen["perm"] == "accept-all"
