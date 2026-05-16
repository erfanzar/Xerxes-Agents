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
"""End-to-end coverage for the multi-step ``/skill-create`` interview.

Drives the daemon's slash-dispatch + ``_submit_turn`` interception path
exactly like the TUI would, and asserts the state machine answers every
question, eventually launches the draft turn, and emits ``turn_end`` on
every intercepted step so the TUI spinner doesn't hang.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from xerxes.daemon.config import DaemonConfig
from xerxes.daemon.runtime import RuntimeManager


class _Recorder:
    """Async EmitFn stand-in that captures every event the daemon emits."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((event_type, payload))

    def slash_bodies(self) -> list[str]:
        return [
            p.get("body", "") for (etype, p) in self.events if etype == "notification" and p.get("category") == "slash"
        ]

    def event_types(self) -> list[str]:
        return [etype for (etype, _) in self.events]


@pytest.fixture
def daemon(tmp_path):
    """Build a ``DaemonServer`` with just enough wiring for the slash path.

    We bypass ``runtime.reload()`` (which needs a real provider) and stub
    out ``_submit_turn`` at the end of the interview so we can assert the
    synthesized prompt without actually running an agent loop.
    """
    from xerxes.daemon.server import DaemonServer

    server = DaemonServer.__new__(DaemonServer)
    server.config = DaemonConfig(project_dir=str(tmp_path))
    server.runtime = RuntimeManager(server.config)
    server.runtime.runtime_config = {"permission_mode": "auto"}
    # Point skills_dir at the tmp_path so we don't pollute the real
    # ``~/.xerxes/skills``.
    server.runtime.skills_dir = tmp_path / "skills"
    server._current_session_key = "tui:default"
    server._current_mode = "code"
    server._current_plan_mode = False
    server._pending_slash_arg = None
    server._pending_skill_create = None
    server.workspaces = type("W", (), {"default_agent_id": "default"})()

    # Capture the synthesized draft prompt instead of running a real turn.
    captured_submit: list[dict[str, Any]] = []
    # Real ``_track_task`` needs an event loop; the test fixture instead
    # collects scheduled coroutines so we can run them on demand.
    pending_post_turn: list[Any] = []
    server._background_tasks = set()

    def _fake_track_task(coro: Any):
        pending_post_turn.append(coro)

        class _DummyTask:
            def done(self) -> bool:
                return True

            def add_done_callback(self, *_a, **_kw) -> None:
                return None

        return _DummyTask()

    server._track_task = _fake_track_task  # type: ignore[assignment]
    server.pending_post_turn = pending_post_turn  # type: ignore[attr-defined]

    async def _fake_submit_turn(params: dict[str, Any], emit) -> dict[str, Any]:
        captured_submit.append(params)
        # Mimic the real ``_submit_turn`` shape so ``_launch_skill_draft``'s
        # post-turn refresh hook still sees a task handle.
        import asyncio as _asyncio

        async def _noop() -> None:
            return None

        try:
            loop = _asyncio.get_running_loop()
            fake_task = loop.create_task(_noop())
        except RuntimeError:
            fake_task = None
        return {"ok": True, "session": {}, "turn_task": fake_task}

    server._submit_turn_real = DaemonServer._submit_turn.__get__(server)
    server._submit_turn = _fake_submit_turn  # type: ignore[assignment]
    server.captured_submit = captured_submit  # type: ignore[attr-defined]
    return server


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _drive_slash(server, command: str) -> _Recorder:
    rec = _Recorder()
    _run(server._handle_slash(command, rec))
    return rec


def _drive_prompt(server, text: str) -> _Recorder:
    """Simulate the TUI sending a chat message — exercises the real
    interception path inside ``_submit_turn``.
    """
    rec = _Recorder()
    _run(server._submit_turn_real({"text": text, "session_key": server._current_session_key}, rec))
    return rec


class TestSkillCreateInterview:
    def test_bare_command_asks_for_slug(self, daemon):
        rec = _drive_slash(daemon, "/skill-create")
        bodies = rec.slash_bodies()
        assert any("What should this skill be called" in b for b in bodies)
        assert daemon._pending_slash_arg == ("skill-create", "tui:default")
        assert daemon._pending_skill_create is None

    def test_slug_then_scope_questions(self, daemon):
        # 1) Bare /skill-create — daemon parks for slug.
        _drive_slash(daemon, "/skill-create")
        # 2) User answers with the slug. This goes through _submit_turn.
        rec = _drive_prompt(daemon, "review-changes")
        # The slug answer should pivot into the scope interview and emit the
        # first question ("what should this skill do?"), plus a bookended
        # turn_begin/turn_end pair so the TUI spinner clears.
        bodies = rec.slash_bodies()
        assert any("What should this skill do" in b for b in bodies), bodies
        assert rec.event_types().count("turn_begin") == 1
        assert rec.event_types().count("turn_end") == 1
        # Interview state should be live and slug-named.
        assert daemon._pending_skill_create is not None
        assert daemon._pending_skill_create["name"] == "review-changes"
        assert daemon._pending_skill_create["answers"] == {}
        # The earlier ``_pending_slash_arg`` should be cleared.
        assert daemon._pending_slash_arg is None

    def test_full_interview_collects_every_answer_and_launches_draft(self, daemon):
        _drive_slash(daemon, "/skill-create")
        _drive_prompt(daemon, "review-changes")  # slug
        rec_what = _drive_prompt(daemon, "Read git diff and find risky changes.")
        assert any("When should a future session activate" in b for b in rec_what.slash_bodies())
        assert daemon._pending_skill_create["answers"]["what"] == "Read git diff and find risky changes."
        # Every intercepted message must emit a turn_end so the TUI spinner clears.
        assert "turn_end" in rec_what.event_types()

        rec_when = _drive_prompt(daemon, "Before any /commit or when I say review-changes.")
        assert any("Which tools or commands" in b for b in rec_when.slash_bodies())
        assert daemon._pending_skill_create["answers"]["when"] == "Before any /commit or when I say review-changes."
        assert "turn_end" in rec_when.event_types()

        rec_tools = _drive_prompt(daemon, "git status, git diff --stat, git diff")
        assert any("Any pitfalls" in b for b in rec_tools.slash_bodies())
        assert daemon._pending_skill_create["answers"]["tools"] == "git status, git diff --stat, git diff"
        assert "turn_end" in rec_tools.event_types()

        rec_pitfalls = _drive_prompt(daemon, "")  # pitfalls is optional — empty is OK
        # All four answers collected — the draft should now be submitted.
        assert daemon._pending_skill_create is None
        assert daemon.captured_submit, "Draft turn was never submitted"
        synthetic = daemon.captured_submit[0]
        assert synthetic.get("_internal_slash") is True
        text = synthetic["text"]
        assert "review-changes" in text
        assert "Read git diff" in text  # what
        assert "Before any /commit" in text  # when
        assert "git status" in text  # tools
        # Pitfalls was empty — the prompt should tell the model to omit that section.
        assert "no pitfalls" in text.lower() or "omit" in text.lower()
        # turn_end always fires on the intercept side.
        assert "turn_end" in rec_pitfalls.event_types()

    def test_cancel_during_interview_clears_state_and_emits_turn_end(self, daemon):
        _drive_slash(daemon, "/skill-create")
        _drive_prompt(daemon, "review-changes")  # slug → first scope question
        rec = _drive_prompt(daemon, "/cancel")
        assert daemon._pending_skill_create is None
        assert daemon.captured_submit == []  # no draft submitted
        bodies = rec.slash_bodies()
        assert any("Cancelled" in b for b in bodies)
        assert "turn_end" in rec.event_types(), "Cancel must still emit turn_end so the TUI spinner clears"

    def test_required_field_rejects_empty_answer(self, daemon):
        _drive_slash(daemon, "/skill-create")
        _drive_prompt(daemon, "review-changes")  # slug
        rec = _drive_prompt(daemon, "   ")  # whitespace-only answer for "what"
        bodies = rec.slash_bodies()
        assert any("required" in b.lower() for b in bodies)
        # We should still be on the same step.
        assert "what" not in daemon._pending_skill_create["answers"]
        assert "turn_end" in rec.event_types()

    def test_invalid_slug_re_prompts_for_name(self, daemon):
        _drive_slash(daemon, "/skill-create")
        rec = _drive_prompt(daemon, "!!!")  # sanitises to empty
        bodies = rec.slash_bodies()
        assert any("doesn't look like a valid slug" in b for b in bodies)
        # The slug arg should be re-parked so the next message tries again.
        assert daemon._pending_slash_arg == ("skill-create", "tui:default")
        assert daemon._pending_skill_create is None
        assert "turn_end" in rec.event_types()

    def test_inline_slug_skips_name_prompt(self, daemon):
        # /skill-create some-name — should jump straight to the scope interview.
        rec = _drive_slash(daemon, "/skill-create commit-helper")
        bodies = rec.slash_bodies()
        assert any("What should this skill do" in b for b in bodies)
        assert daemon._pending_skill_create is not None
        assert daemon._pending_skill_create["name"] == "commit-helper"
        # No name prompt was emitted.
        assert not any("What should this skill be called" in b for b in bodies)

    def test_auto_at_first_question_fast_forwards_to_draft(self, daemon):
        _drive_slash(daemon, "/skill-create commit-helper")
        rec = _drive_prompt(daemon, "auto")
        # All four fields are auto, so the interview ends and the draft is
        # submitted immediately with every key marked _auto.
        assert daemon._pending_skill_create is None
        assert daemon.captured_submit, "Draft turn was never submitted"
        text = daemon.captured_submit[0]["text"]
        # The synthesized prompt must mark every field as auto so the model
        # knows to infer it.
        assert text.count("_auto") >= 3  # what / when / tools at minimum
        assert "infer" in text.lower() or "session" in text.lower()
        # Announcement mentions the auto-inferred keys.
        bodies = rec.slash_bodies()
        assert any("inferring" in b for b in bodies)
        assert "turn_end" in rec.event_types()

    def test_auto_mid_interview_keeps_prior_answers(self, daemon):
        _drive_slash(daemon, "/skill-create commit-helper")
        _drive_prompt(daemon, "Summarise the diff and flag risky bits.")  # what
        _drive_prompt(daemon, "auto")  # auto-fill when, tools, pitfalls
        assert daemon._pending_skill_create is None
        assert daemon.captured_submit, "Draft turn was never submitted"
        text = daemon.captured_submit[0]["text"]
        assert "Summarise the diff and flag risky bits." in text
        # The remaining three fields should be rendered as _auto_.
        assert text.count("_auto") >= 3

    def test_post_draft_refresh_emits_init_done(self, daemon, monkeypatch):
        """After the draft turn finishes, the TUI's autocomplete cache is
        refreshed by an ``init_done`` event listing the latest skills.
        """
        # Stub ``discover_skills`` so we can verify the refresh runs.
        discovered: list[dict[str, str]] = [
            {"name": "commit-helper", "description": "Auto-drafted"},
        ]
        monkeypatch.setattr(daemon.runtime, "discover_skills", lambda: discovered)

        _drive_slash(daemon, "/skill-create commit-helper")
        rec = _drive_prompt(daemon, "auto")  # full auto — fastest path to launch

        # The draft turn was submitted and a post-turn coroutine was scheduled.
        assert daemon.captured_submit
        assert daemon.pending_post_turn, "Post-turn refresh coroutine was not scheduled"

        # Run the scheduled post-turn coroutine and assert init_done fires.
        _run(daemon.pending_post_turn[0])
        init_done_events = [(etype, payload) for (etype, payload) in rec.events if etype == "init_done"]
        assert init_done_events, "init_done was not emitted after the draft turn"
        assert init_done_events[0][1].get("skills") == discovered
