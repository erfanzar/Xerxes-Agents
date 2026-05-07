from __future__ import annotations

import pytest
from xerxes.tui.app import XerxesTUI


class _ClientStub:
    def __init__(self) -> None:
        self.cancelled = 0
        self.cancelled_all = 0

    async def cancel(self) -> None:
        self.cancelled += 1

    async def cancel_all(self) -> None:
        self.cancelled_all += 1


class _PromptStub:
    def __init__(self) -> None:
        self.running: bool | None = None
        self.lines: list[str] = []
        self.cleared_tools = False
        self.cleared_previews = False

    def clear_active_approval(self) -> None:
        pass

    def clear_active_question(self) -> None:
        pass

    def clear_active_tools(self) -> None:
        self.cleared_tools = True

    def clear_subagent_previews(self) -> None:
        self.cleared_previews = True

    def clear_thinking(self) -> None:
        pass

    def set_running(self, running: bool) -> None:
        self.running = running

    def append_line(self, line: str) -> None:
        self.lines.append(line)


@pytest.mark.asyncio
async def test_interrupt_current_turn_clears_ui_and_releases_waiter() -> None:
    tui = XerxesTUI()
    client = _ClientStub()
    prompt = _PromptStub()
    tui._client = client  # type: ignore[assignment]
    tui._prompt = prompt  # type: ignore[assignment]
    tui._turn_done_event.clear()
    restarted = False

    async def restart() -> None:
        nonlocal restarted
        restarted = True

    tui._restart_bridge_after_interrupt = restart  # type: ignore[method-assign]

    await tui._interrupt_current_turn()

    assert client.cancelled_all == 1
    assert prompt.running is False
    assert prompt.cleared_tools is True
    assert prompt.cleared_previews is True
    assert tui._turn_done_event.is_set()
    assert "Interrupted" in prompt.lines[-1]
    assert restarted is True
