from __future__ import annotations

import pytest

from xerxes.tui.app import XerxesTUI
from xerxes.tui.prompt import PersistentPrompt
from xerxes.streaming.wire_events import StatusUpdate


class _ClientStub:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict]] = []
        self.steered: list[str] = []
        self.queries: list[tuple[str, bool]] = []
        self.tui: XerxesTUI | None = None

    async def _send_jsonrpc(self, method: str, params: dict, req_id: str | None = None) -> None:
        self.sent.append((method, params))

    async def steer(self, content: str) -> None:
        self.steered.append(content)

    async def query(self, user_input: str, plan_mode: bool = False) -> None:
        self.queries.append((user_input, plan_mode))
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
async def test_shift_tab_plan_toggle_updates_ui_and_bridge() -> None:
    tui = XerxesTUI()
    client = _ClientStub()
    prompt = _PromptStub()
    tui._client = client  # type: ignore[assignment]
    tui._prompt = prompt  # type: ignore[assignment]

    await tui._toggle_plan_mode()

    assert tui._plan_mode is True
    assert prompt.plan_mode is True
    assert client.sent == [("set_plan_mode", {"enabled": True})]
    assert "Plan mode ON" in prompt.lines[-1]


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
    assert ("set_plan_mode", {"enabled": True}) in client.sent
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

    assert client.queries == [("write a plan", True)]
    assert tui._plan_mode is False
    assert prompt.plan_mode is False
    assert prompt.activity_mode == "code"
    assert ("set_plan_mode", {"enabled": False}) in client.sent
    assert "Code mode ON" in prompt.lines[-1]


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
