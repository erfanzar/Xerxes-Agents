from __future__ import annotations

from xerxes.tui.app import XerxesTUI
from xerxes.tui.blocks import ApprovalRequestPanel


class _PromptStub:
    APPROVAL_SENTINEL = "\x00__select_active_approval__\x00"

    def __init__(self) -> None:
        self.lines: list[str] = []

    def append_line(self, line: str) -> None:
        self.lines.append(line)


def _tui_with_approval() -> tuple[XerxesTUI, _PromptStub, ApprovalRequestPanel]:
    tui = XerxesTUI()
    prompt = _PromptStub()
    panel = ApprovalRequestPanel(
        request_id="req_1",
        tool_call_id="tool_1",
        action="ExecuteShell",
        description="Run: cd /tmp",
    )
    tui._prompt = prompt  # type: ignore[assignment]
    tui._approval_panel = panel
    return tui, prompt, panel


def test_approval_sentinel_uses_selected_response() -> None:
    tui, prompt, panel = _tui_with_approval()
    panel.move_cursor_down()

    assert tui._resolve_approval_input(prompt.APPROVAL_SENTINEL) == "approve_for_session"


def test_approval_keyboard_aliases() -> None:
    tui, _, _ = _tui_with_approval()

    assert tui._resolve_approval_input("a") == "approve_for_session"
    assert tui._resolve_approval_input("r") == "reject"


def test_invalid_approval_input_keeps_waiting() -> None:
    tui, prompt, _ = _tui_with_approval()

    assert tui._resolve_approval_input("maybe") is None
    assert prompt.lines


def test_approval_panel_outputs_ansi_not_rich_markup() -> None:
    _, _, panel = _tui_with_approval()

    rendered = panel.compose()

    assert "\x1b[" in rendered
    assert "[green]" not in rendered
    assert "[cyan]" not in rendered
    assert "[red]" not in rendered
