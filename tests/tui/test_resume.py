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
"""TUI resume replay handling."""

from __future__ import annotations

from xerxes.runtime.update import GitUpdateStatus
from xerxes.streaming.wire_events import Notification
from xerxes.tui.app import XerxesTUI, _build_welcome_banner
from xerxes.tui.blocks import ResumeSessionPanel
from xerxes.tui.prompt import PersistentPrompt


class _PromptStub:
    RESUME_SENTINEL = "\x00__select_active_resume__\x00"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.resume_panel = None

    def clear_content(self) -> None:
        self.calls.append("clear_content")

    def clear_streaming(self) -> None:
        self.calls.append("clear_streaming")

    def clear_thinking(self) -> None:
        self.calls.append("clear_thinking")

    def clear_active_tools(self) -> None:
        self.calls.append("clear_active_tools")

    def clear_subagent_previews(self) -> None:
        self.calls.append("clear_subagent_previews")

    def append_line(self, _line: str) -> None:
        self.calls.append("append_line")

    def set_active_resume(self, panel) -> None:
        self.calls.append("set_active_resume")
        self.resume_panel = panel

    def clear_active_resume(self) -> None:
        self.calls.append("clear_active_resume")
        self.resume_panel = None

    def refresh_active_resume(self) -> None:
        self.calls.append("refresh_active_resume")


def test_resume_begin_clears_visible_transcript_without_rendering_notification() -> None:
    tui = XerxesTUI()
    prompt = _PromptStub()
    tui._prompt = prompt  # type: ignore[assignment]
    tui._notification_history.append(object())  # type: ignore[arg-type]
    tui._content_blocks["old"] = object()  # type: ignore[assignment]
    tui._tool_blocks["old"] = object()  # type: ignore[assignment]

    tui._on_notification(
        Notification(
            id="resume",
            category="history",
            type="resume_begin",
            body="switching",
        )
    )

    assert prompt.calls == [
        "clear_content",
        "clear_streaming",
        "clear_thinking",
        "clear_active_tools",
        "clear_subagent_previews",
    ]
    assert tui._notification_history == []
    assert tui._content_blocks == {}
    assert tui._tool_blocks == {}


def test_resume_begin_preserves_welcome_banner() -> None:
    tui = XerxesTUI()
    prompt = PersistentPrompt()
    banner = _build_welcome_banner(model="kimi-for-coding", session_id="old-session", cwd="/tmp/xerxes")
    tui._prompt = prompt
    tui._banner_start = len(prompt._status._content_lines)
    prompt.append_line(banner)
    tui._banner_line_count = len(prompt._status._content_lines) - tui._banner_start
    prompt.append_line("old chat line")

    tui._clear_visible_transcript()

    history = "\n".join(prompt._status._content_lines)
    assert "old-session" in history
    assert "kimi-for-coding" in history
    assert "old chat line" not in history
    assert tui._banner_start == 0
    assert tui._banner_line_count == len(prompt._status._content_lines)


def test_welcome_banner_includes_version_head_and_updates() -> None:
    banner = _build_welcome_banner(
        model="kimi-for-coding",
        session_id="session-1",
        cwd="/tmp/xerxes",
        version="0.2.5",
        git_status=GitUpdateStatus(
            is_git=True,
            branch="main",
            upstream="origin/main",
            head_hash="abc1234",
            upstream_hash="def5678",
            behind_count=4,
        ),
    )

    assert "Version:" in banner
    assert "v0.2.5" in banner
    assert "HEAD:" in banner
    assert "abc1234" in banner
    assert "4 ahead available (origin/main def5678)" in banner


def test_resume_choices_open_interactive_picker() -> None:
    tui = XerxesTUI()
    prompt = _PromptStub()
    tui._prompt = prompt  # type: ignore[assignment]

    tui._on_notification(
        Notification(
            id="resume-list",
            category="history",
            type="resume_choices",
            payload={
                "sessions": [
                    {
                        "session_id": "abcd1234",
                        "title": "saved question",
                        "updated_at": "2026-06-19T14:39:27+00:00",
                        "turn_count": 1,
                        "messages": 2,
                    }
                ]
            },
        )
    )

    assert prompt.calls == ["set_active_resume"]
    assert isinstance(tui._resume_panel, ResumeSessionPanel)
    assert prompt.resume_panel is tui._resume_panel
    assert tui._resume_panel.selected_session_id == "abcd1234"


def test_resume_replay_renders_as_transcript_not_notification() -> None:
    tui = XerxesTUI()
    prompt = PersistentPrompt()
    tui._prompt = prompt

    tui._on_notification(
        Notification(
            id="user",
            category="history",
            type="replay_user",
            body="✨ hi",
        )
    )
    tui._on_notification(
        Notification(
            id="assistant",
            category="history",
            type="replay_assistant",
            body="Hello again",
        )
    )
    tui._on_notification(
        Notification(
            id="done",
            category="history",
            type="resumed",
            body="── resumed session abcd1234 (2 messages) ──",
        )
    )

    history = "\n".join(prompt._status._content_lines)
    assert "✨ hi" in history
    assert "Hello again" in history
    assert "resumed session abcd1234" in history
    assert tui._notification_history == []


def test_resume_replay_renders_assistant_markdown() -> None:
    tui = XerxesTUI()
    prompt = PersistentPrompt()
    tui._prompt = prompt

    tui._on_notification(
        Notification(
            id="assistant",
            category="history",
            type="replay_assistant",
            body="**Hello**\n\n- item",
        )
    )

    history = "\n".join(prompt._status._content_lines)
    assert "**Hello**" not in history
    assert "\x1b[1mHello" in history
    assert "item" in history


def test_resume_picker_resolves_enter_and_numbers() -> None:
    tui = XerxesTUI()
    prompt = _PromptStub()
    tui._prompt = prompt  # type: ignore[assignment]
    tui._resume_panel = ResumeSessionPanel(
        [
            {"session_id": "first", "title": "first prompt"},
            {"session_id": "second", "title": "second prompt"},
        ]
    )

    assert tui._resolve_resume_input("2") == "second"
    assert tui._resolve_resume_input(prompt.RESUME_SENTINEL) == "first"
    assert tui._resolve_resume_input("second prompt") == "second prompt"
    assert tui._resolve_resume_input("/cancel") == ""
