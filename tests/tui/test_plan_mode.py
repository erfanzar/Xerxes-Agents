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

from __future__ import annotations

import pytest
from prompt_toolkit.data_structures import Point
from prompt_toolkit.keys import Keys
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from xerxes.streaming.wire_events import InitDone, StatusUpdate
from xerxes.tui.app import XerxesTUI, _build_welcome_banner
from xerxes.tui.prompt import FooterRenderer, PersistentPrompt, StatusRenderer


class _ClientStub:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict]] = []
        self.steered: list[str] = []
        self.queries: list[tuple[str, bool, str]] = []
        self.tui: XerxesTUI | None = None

    async def _send_jsonrpc(self, method: str, params: dict, req_id: str | None = None) -> None:
        self.sent.append((method, params))

    async def steer(self, content: str) -> None:
        self.steered.append(content)

    async def query(self, user_input: str, plan_mode: bool = False, mode: str = "code") -> None:
        self.queries.append((user_input, plan_mode, mode))
        if self.tui is not None:
            self.tui._turn_done_event.set()


class _PromptStub:
    PLAN_TOGGLE_SENTINEL = PersistentPrompt.PLAN_TOGGLE_SENTINEL

    def __init__(self) -> None:
        self.plan_mode: bool | None = None
        self.activity_mode: str | None = None
        self.lines: list[str] = []

    def set_plan_mode(self, enabled: bool) -> None:
        self.plan_mode = enabled

    def set_activity_mode(self, mode: str) -> None:
        self.activity_mode = mode

    def set_reasoning_effort(self, effort: str) -> None:
        self.reasoning_effort = effort

    def append_line(self, line: str) -> None:
        self.lines.append(line)

    def set_context(self, _used: int, _max: int) -> None:
        pass

    def set_running(self, _running: bool) -> None:
        pass

    def set_queue_count(self, _count: int) -> None:
        pass

    def clear_active_approval(self) -> None:
        pass


def test_tui_defaults_to_accept_all_permissions() -> None:
    assert XerxesTUI()._permission_mode == "accept-all"


@pytest.mark.asyncio
async def test_shift_tab_cycles_code_plan_research_objective_modes() -> None:
    tui = XerxesTUI()
    client = _ClientStub()
    prompt = _PromptStub()
    tui._client = client  # type: ignore[assignment]
    tui._prompt = prompt  # type: ignore[assignment]

    await tui._cycle_interaction_mode()

    assert tui._plan_mode is True
    assert prompt.plan_mode is True
    assert client.sent == [("set_plan_mode", {"enabled": True, "mode": "plan"})]
    assert prompt.lines == []

    await tui._cycle_interaction_mode()

    assert tui._plan_mode is False
    assert prompt.plan_mode is False
    assert prompt.activity_mode == "researcher"
    assert client.sent[-1] == ("set_plan_mode", {"enabled": False, "mode": "researcher"})

    await tui._cycle_interaction_mode()

    assert tui._plan_mode is False
    assert prompt.plan_mode is False
    assert prompt.activity_mode == "objective"
    assert client.sent[-1] == ("set_mode", {"mode": "objective"})

    await tui._cycle_interaction_mode()

    assert tui._plan_mode is False
    assert prompt.plan_mode is False
    assert prompt.activity_mode == "code"
    assert client.sent[-1] == ("set_mode", {"mode": "code"})


@pytest.mark.asyncio
async def test_slash_plan_uses_same_plan_mode_state_path() -> None:
    tui = XerxesTUI()
    client = _ClientStub()
    prompt = _PromptStub()
    tui._client = client  # type: ignore[assignment]
    tui._prompt = prompt  # type: ignore[assignment]

    await tui._handle_slash("/plan inspect auth")

    assert tui._plan_mode is True
    assert prompt.plan_mode is True
    assert ("set_plan_mode", {"enabled": True, "mode": "plan"}) in client.sent
    assert client.steered == ["/plan inspect auth"]


@pytest.mark.asyncio
async def test_slash_objective_sets_sticky_objective_mode() -> None:
    tui = XerxesTUI()
    client = _ClientStub()
    prompt = _PromptStub()
    tui._client = client  # type: ignore[assignment]
    tui._prompt = prompt  # type: ignore[assignment]

    await tui._handle_slash("/objective beat benchmark")

    assert tui._plan_mode is False
    assert prompt.plan_mode is False
    assert prompt.activity_mode == "objective"
    assert tui._user_activity_mode == "objective"
    assert ("set_plan_mode", {"enabled": False, "mode": "objective"}) in client.sent
    assert client.steered == ["/objective beat benchmark"]


def test_status_update_syncs_plan_mode_to_ui() -> None:
    tui = XerxesTUI()
    prompt = _PromptStub()
    tui._prompt = prompt  # type: ignore[assignment]

    tui._on_status_update(StatusUpdate(context_tokens=1, max_context=100, plan_mode=True))

    assert tui._plan_mode is True
    assert prompt.plan_mode is True


@pytest.mark.asyncio
async def test_plan_mode_turn_auto_returns_to_code_mode() -> None:
    tui = XerxesTUI()
    client = _ClientStub()
    prompt = _PromptStub()
    client.tui = tui
    tui._client = client  # type: ignore[assignment]
    tui._prompt = prompt  # type: ignore[assignment]
    tui._plan_mode = True
    prompt.plan_mode = True

    await tui._run_turns("write a plan")

    assert client.queries == [("write a plan", True, "plan")]
    assert tui._plan_mode is False
    assert prompt.plan_mode is False
    assert prompt.activity_mode == "code"
    assert ("set_plan_mode", {"enabled": False, "mode": "code"}) in client.sent
    assert prompt.lines == []


@pytest.mark.asyncio
async def test_user_selected_research_mode_survives_turn_start() -> None:
    tui = XerxesTUI()
    client = _ClientStub()
    prompt = _PromptStub()
    client.tui = tui
    tui._client = client  # type: ignore[assignment]
    tui._prompt = prompt  # type: ignore[assignment]

    await tui._cycle_interaction_mode()
    await tui._cycle_interaction_mode()
    await tui._run_turns("research this")

    assert client.queries == [("research this", False, "researcher")]
    assert tui._plan_mode is False
    assert prompt.activity_mode == "researcher"


def test_activity_mode_infers_researcher_and_coder_from_tools() -> None:
    assert XerxesTUI._infer_activity_mode("ReadFile", "{}") == "researcher"
    assert XerxesTUI._infer_activity_mode("GrepTool", "{}") == "researcher"
    assert XerxesTUI._infer_activity_mode("WriteFile", "{}") == "code"
    assert XerxesTUI._infer_activity_mode("FileEditTool", "{}") == "code"


def test_activity_mode_infers_subagent_type_from_agent_tool() -> None:
    assert XerxesTUI._infer_activity_mode("AgentTool", '{"subagent_type":"researcher"}') == "researcher"
    assert XerxesTUI._infer_activity_mode("AgentTool", '{"subagent_type":"coder"}') == "coder"
    assert (
        XerxesTUI._infer_activity_mode(
            "SpawnAgents",
            '{"agents":[{"subagent_type":"researcher"},{"subagent_type":"researcher"}]}',
        )
        == "researcher"
    )
    assert (
        XerxesTUI._infer_activity_mode(
            "SpawnAgents",
            '{"agents":[{"subagent_type":"researcher"},{"subagent_type":"coder"}]}',
        )
        == "agents"
    )


def test_status_update_without_mode_keeps_non_plan_activity_mode() -> None:
    tui = XerxesTUI()
    prompt = _PromptStub()
    tui._prompt = prompt  # type: ignore[assignment]

    tui._set_activity_mode("researcher")
    tui._on_status_update(StatusUpdate(context_tokens=1, max_context=100, plan_mode=False, mode=""))

    assert tui._plan_mode is False
    assert prompt.activity_mode == "researcher"


def test_status_update_accepts_explicit_code_mode() -> None:
    tui = XerxesTUI()
    prompt = _PromptStub()
    tui._prompt = prompt  # type: ignore[assignment]

    tui._set_activity_mode("researcher", user_selected=True)
    tui._on_status_update(StatusUpdate(context_tokens=1, max_context=100, plan_mode=False, mode="code"))

    assert tui._plan_mode is False
    assert prompt.activity_mode == "code"
    assert tui._user_activity_mode is None


def test_status_update_makes_model_selected_objective_mode_sticky() -> None:
    tui = XerxesTUI()
    prompt = _PromptStub()
    tui._prompt = prompt  # type: ignore[assignment]

    tui._on_status_update(StatusUpdate(context_tokens=1, max_context=100, plan_mode=False, mode="objective"))

    assert tui._plan_mode is False
    assert prompt.activity_mode == "objective"
    assert tui._user_activity_mode == "objective"


def test_plan_mode_colors_footer_and_input_separator_purple() -> None:
    from xerxes.tui.skin_engine import active_fg

    plan_color = active_fg("system")  # plan mode tints via the skin "system" (violet) role

    footer = FooterRenderer()
    footer.set_plan_mode(True)

    footer_markup = footer._markup()
    assert plan_color in footer_markup
    assert "mode: plan" in footer_markup

    status = StatusRenderer()
    status.set_plan_mode(True)

    status_markup = status._markup()
    assert plan_color in status_markup
    assert "input · plan" in status_markup


def test_research_mode_colors_footer_and_input_separator_cyan() -> None:
    from xerxes.tui.skin_engine import active_fg

    research_color = active_fg("accent")  # researcher mode tints via the skin "accent" (turquoise) role

    footer = FooterRenderer()
    footer.set_activity_mode("researcher")

    footer_markup = footer._markup()
    assert research_color in footer_markup
    assert "mode: researcher" in footer_markup

    status = StatusRenderer()
    status.set_activity_mode("researcher")

    status_markup = status._markup()
    assert research_color in status_markup
    assert "input · research" in status_markup


def test_objective_mode_colors_footer_and_input_separator_yellow() -> None:
    from xerxes.tui.skin_engine import active_fg

    objective_color = active_fg("warn")

    footer = FooterRenderer()
    footer.set_activity_mode("objective")

    footer_markup = footer._markup()
    assert objective_color in footer_markup
    assert "mode: objective" in footer_markup

    status = StatusRenderer()
    status.set_activity_mode("objective")

    status_markup = status._markup()
    assert objective_color in status_markup
    assert "input · objective" in status_markup


def test_status_renderer_trims_thinking_preview_to_last_non_empty_lines() -> None:
    status = StatusRenderer()

    status.append_thinking("\n\nfirst\n\nsecond\nthird\nfourth\nfifth\n")

    markup = status._markup()
    assert "\n\n✻" not in markup
    assert "first" not in markup
    assert "second\nthird\nfourth\nfifth" in markup


def test_status_renderer_trims_blank_edges_from_committed_lines() -> None:
    status = StatusRenderer()

    status.append_line("\n\nhello\n\n")

    markup = status._markup()
    assert "\n\nhello" not in markup
    assert "hello\n" in markup


def test_status_renderer_agent_dashboard_caps_rows() -> None:
    status = StatusRenderer()

    n = StatusRenderer.AGENT_DASHBOARD_MAX_ROWS + 4
    for idx in range(n):
        status.set_subagent_preview(f"task-{idx}", f"agent-{idx}", f"Tool{idx}")

    markup = status._markup()
    assert f"{n} agents" in markup  # aggregate header shows the total
    assert "+4 more" in markup  # overflow beyond the cap is collapsed
    assert "agent-0" in markup  # first agents are shown
    assert f"agent-{n - 1}" not in markup  # last agents fall under the cap


def test_status_cursor_uses_current_render_metrics_for_dynamic_content() -> None:
    prompt = PersistentPrompt()
    calls = 0

    def render_tool() -> str:
        nonlocal calls
        calls += 1
        return "first\nsecond" if calls == 1 else "first"

    prompt._status.set_active_tool("dynamic", render_tool)

    content = prompt._status_control.create_content(width=80, height=None)

    assert content.cursor_position is not None
    assert content.cursor_position.y < content.line_count
    assert calls == 1


def test_prompt_scroll_uses_rendered_lines_and_visible_height() -> None:
    prompt = PersistentPrompt()
    prompt._set_status_visible_rows(6)
    for idx in range(30):
        prompt.append_line(f"line {idx}")

    prompt._scroll_page(-1)

    assert prompt._scroll_y == 20

    markup = prompt._status._markup(prompt._scroll_y, visible_rows=prompt._status_visible_rows)
    assert "line 20" in markup
    assert "line 24" in markup
    assert "line 25" not in markup

    prompt._scroll_page(1)

    assert prompt._scroll_y is None


def test_prompt_scroll_key_aliases_are_registered() -> None:
    prompt = PersistentPrompt()
    keys = {binding.keys for binding in prompt._kb.bindings}

    assert (Keys.PageUp,) in keys
    assert (Keys.ShiftPageUp,) in keys
    assert (Keys.ControlPageUp,) in keys
    assert (Keys.PageDown,) in keys
    assert (Keys.ShiftPageDown,) in keys
    assert (Keys.ControlPageDown,) in keys
    assert (Keys.ControlUp,) in keys
    assert (Keys.ControlDown,) in keys
    assert (Keys.ScrollUp,) in keys
    assert (Keys.ScrollDown,) in keys


def test_mouse_wheel_over_input_routes_to_transcript_scroll() -> None:
    prompt = PersistentPrompt()
    prompt._set_status_visible_rows(6)
    for idx in range(30):
        prompt.append_line(f"line {idx}")

    event = MouseEvent(
        position=Point(x=0, y=0),
        event_type=MouseEventType.SCROLL_UP,
        button=MouseButton.NONE,
        modifiers=frozenset(),
    )

    assert prompt._buffer_control.mouse_handler(event) is None
    assert prompt._scroll_y == 24


@pytest.mark.asyncio
async def test_init_done_replaces_full_split_banner_range() -> None:
    tui = XerxesTUI()
    prompt = PersistentPrompt()
    cwd = "/tmp/xerxes"
    provisional_banner = _build_welcome_banner(model="", session_id="provisional", cwd=cwd)

    tui._prompt = prompt
    tui._banner_cwd = cwd
    tui._banner_start = len(prompt._status._content_lines)
    prompt.append_line(provisional_banner)
    tui._banner_line_count = len(prompt._status._content_lines) - tui._banner_start

    tui._on_init_done(InitDone(model="glm-5.2", session_id="real-session", cwd=cwd))
    if tui._model_load_task is not None:
        await tui._model_load_task

    history = "\n".join(prompt._status._content_lines)
    from xerxes.tui.skin_engine import get_active_skin

    welcome = get_active_skin().label("welcome")
    assert history.count(welcome) == 1
    assert "provisional" not in history
    assert "real-session" in history
    assert "glm-5.2" in history
