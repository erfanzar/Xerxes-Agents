from __future__ import annotations

import pytest
from xerxes.streaming.wire_events import StatusUpdate
from xerxes.tui.app import XerxesTUI
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


@pytest.mark.asyncio
async def test_shift_tab_cycles_code_plan_research_modes() -> None:
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
    assert prompt.activity_mode == "code"


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


def test_status_update_keeps_non_plan_activity_mode() -> None:
    tui = XerxesTUI()
    prompt = _PromptStub()
    tui._prompt = prompt  # type: ignore[assignment]

    tui._set_activity_mode("researcher")
    tui._on_status_update(StatusUpdate(context_tokens=1, max_context=100, plan_mode=False))

    assert tui._plan_mode is False
    assert prompt.activity_mode == "researcher"


def test_plan_mode_colors_footer_and_input_separator_purple() -> None:
    footer = FooterRenderer()
    footer.set_plan_mode(True)

    footer_markup = footer._markup()
    assert "\x1b[35m" in footer_markup
    assert "mode: plan" in footer_markup

    status = StatusRenderer()
    status.set_plan_mode(True)

    status_markup = status._markup()
    assert "\x1b[35m" in status_markup
    assert "input · plan" in status_markup


def test_research_mode_colors_footer_and_input_separator_cyan() -> None:
    footer = FooterRenderer()
    footer.set_activity_mode("researcher")

    footer_markup = footer._markup()
    assert "\x1b[36m" in footer_markup
    assert "mode: researcher" in footer_markup

    status = StatusRenderer()
    status.set_activity_mode("researcher")

    status_markup = status._markup()
    assert "\x1b[36m" in status_markup
    assert "input · research" in status_markup


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


def test_status_renderer_shows_last_five_subagent_previews() -> None:
    status = StatusRenderer()

    for idx in range(8):
        status.set_subagent_preview(f"task-{idx}", f"agent-{idx}", f"Tool{idx}")

    markup = status._markup()
    for idx in range(3):
        assert f"agent-{idx}" not in markup
    for idx in range(3, 8):
        assert f"agent-{idx}" in markup


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
