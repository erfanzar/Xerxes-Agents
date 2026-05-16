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
"""Daemon resumes loaded session messages into the TUI's scrollback.

Persistence (test_session_persistence) verified the bytes land on disk
and are loaded into ``state.messages``. This file pins the next
contract: the daemon emits one ``notification`` per user/assistant
turn so the TUI scrollback isn't blank after ``xerxes -r <id>``."""

from __future__ import annotations

import asyncio

from xerxes.daemon.config import DaemonConfig
from xerxes.daemon.runtime import RuntimeManager, WorkspaceManager
from xerxes.daemon.server import DaemonServer


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def __call__(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))

    def histories(self) -> list[dict]:
        return [
            payload
            for (etype, payload) in self.events
            if etype == "notification" and payload.get("category") == "history"
        ]


def _make_server(tmp_path) -> DaemonServer:
    """Construct a minimally-wired DaemonServer for slash/replay tests."""
    server = DaemonServer.__new__(DaemonServer)
    config = DaemonConfig(project_dir=str(tmp_path))
    config.workspace = {"root": str(tmp_path / "agents"), "default_agent_id": "default"}
    server.config = config
    server.workspaces = WorkspaceManager(config)
    server.runtime = RuntimeManager(config)
    server.runtime.runtime_config = {"permission_mode": "auto"}
    from xerxes.daemon.runtime import SessionManager

    server.sessions = SessionManager(server.workspaces, store_dir=tmp_path / "sessions")
    return server


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class TestRenderMessageText:
    def test_string(self):
        assert DaemonServer._render_message_text("hello") == "hello"

    def test_list_of_parts(self):
        out = DaemonServer._render_message_text([{"text": "a"}, {"text": "b"}])
        assert out == "a\nb"

    def test_dict(self):
        assert DaemonServer._render_message_text({"text": "x"}) == "x"

    def test_none(self):
        assert DaemonServer._render_message_text(None) == ""

    def test_strips_whitespace(self):
        assert DaemonServer._render_message_text("  hi  \n") == "hi"


class TestSkillActivationDetection:
    def test_classic_marker(self):
        assert DaemonServer._looks_like_skill_activation("[Skill 'foo' activated]\n\nbody")

    def test_with_leading_whitespace(self):
        assert DaemonServer._looks_like_skill_activation("   [Skill 'autoresearch' activated]")

    def test_regular_user_message(self):
        assert not DaemonServer._looks_like_skill_activation("Hello, can you help me?")

    def test_almost_match(self):
        # No "activated" → not a skill prompt.
        assert not DaemonServer._looks_like_skill_activation("[Skill 'foo' something else]")


class TestReplaySessionHistory:
    def test_replays_user_and_assistant_turns(self, tmp_path):
        server = _make_server(tmp_path)
        session = server.sessions.open("abc12345")
        session.state.messages = [
            {"role": "user", "content": "what is xerxes?"},
            {"role": "assistant", "content": "Xerxes is a multi-agent framework."},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "anytime"},
        ]

        recorder = _Recorder()
        _run(server._replay_session_history(session, recorder))
        bodies = [h["body"] for h in recorder.histories()]
        assert any("what is xerxes?" in b for b in bodies)
        assert any("multi-agent framework" in b for b in bodies)
        assert any("anytime" in b for b in bodies)
        # Final summary line with count.
        summary = [h for h in recorder.histories() if h["type"] == "resumed"]
        assert summary and "4 messages" in summary[0]["body"]

    def test_skips_tool_role_messages(self, tmp_path):
        server = _make_server(tmp_path)
        session = server.sessions.open("abc12345")
        session.state.messages = [
            {"role": "user", "content": "go"},
            {"role": "tool", "content": "tool result is huge"},
            {"role": "assistant", "content": "done"},
        ]
        recorder = _Recorder()
        _run(server._replay_session_history(session, recorder))
        types = [h["type"] for h in recorder.histories() if h["type"].startswith("replay_")]
        # Two replay entries (user + assistant); tool not surfaced.
        assert types == ["replay_user", "replay_assistant"]

    def test_skips_skill_activation_prompts(self, tmp_path):
        server = _make_server(tmp_path)
        session = server.sessions.open("abc12345")
        session.state.messages = [
            {"role": "user", "content": "[Skill 'autoresearch' activated]\n\n## Skill body…"},
            {"role": "assistant", "content": "Running autoresearch…"},
            {"role": "user", "content": "actually use the fix variant"},
            {"role": "assistant", "content": "Switching to /autoresearch:fix"},
        ]
        recorder = _Recorder()
        _run(server._replay_session_history(session, recorder))
        bodies = [h["body"] for h in recorder.histories() if h["type"].startswith("replay_")]
        # Skill activation filtered; the rest replayed.
        assert not any("Skill 'autoresearch' activated" in b for b in bodies)
        assert any("actually use the fix variant" in b for b in bodies)
        assert any("Switching to /autoresearch:fix" in b for b in bodies)

    def test_empty_session_just_emits_summary(self, tmp_path):
        server = _make_server(tmp_path)
        session = server.sessions.open("abc12345")
        session.state.messages = []
        recorder = _Recorder()
        _run(server._replay_session_history(session, recorder))
        histories = recorder.histories()
        assert len(histories) == 1
        assert histories[0]["type"] == "resumed"
        assert "0 messages" in histories[0]["body"]

    def test_user_messages_prefixed_with_marker(self, tmp_path):
        server = _make_server(tmp_path)
        session = server.sessions.open("abc12345")
        session.state.messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
        recorder = _Recorder()
        _run(server._replay_session_history(session, recorder))
        bodies = [h["body"] for h in recorder.histories() if h["type"] == "replay_user"]
        assert bodies[0].startswith("✨")


class TestInitializeReplaysWhenResuming:
    """End-to-end: initialize() with a resumed session emits history events."""

    def test_resume_with_history_replays(self, tmp_path):
        server = _make_server(tmp_path)
        # Pre-seed: save a session.
        sess = server.sessions.open("abcd1234")
        sess.state.messages = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ]
        sess.state.turn_count = 1
        server.sessions.save(sess)
        # Wipe in-memory state so initialize() takes the load path.
        server.sessions._sessions.clear()
        # Stub the bits initialize() touches that we don't care about.
        server.runtime.reload = lambda overrides=None: None
        server.runtime.discover_skills = lambda: []
        server.runtime.runtime_config = {"permission_mode": "auto", "model": "claude-haiku-4-5"}
        server.runtime.skill_registry = type(server.runtime.skill_registry)()
        # Required for /context handler used in _emit_status — we stub.
        server._git_branch = lambda: ""
        server._emit_status = _stub_async  # type: ignore[method-assign]

        recorder = _Recorder()
        _run(server._initialize({"resume_session_id": "abcd1234"}, recorder))

        # init_done emitted first.
        assert any(etype == "init_done" for etype, _ in recorder.events)
        # And the prior turns are replayed as history notifications.
        bodies = [h["body"] for h in recorder.histories() if h["type"].startswith("replay_")]
        assert any("first question" in b for b in bodies)
        assert any("first answer" in b for b in bodies)


async def _stub_async(*a, **kw) -> None:
    return None
