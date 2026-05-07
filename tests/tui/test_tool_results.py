from __future__ import annotations

from xerxes.tui.app import XerxesTUI


class _PromptStub:
    def __init__(self) -> None:
        self.committed: list[tuple[str, str]] = []

    def commit_streaming(self) -> None:
        pass

    def set_spinner_label(self, _label: str) -> None:
        pass

    def reset_spinner_timer(self) -> None:
        pass

    def set_active_tool(self, _tool_call_id, _render_fn) -> None:
        pass

    def commit_active_tool(self, tool_call_id: str, final_text: str) -> None:
        self.committed.append((tool_call_id, final_text))


def test_agent_tool_result_is_not_rendered_in_tui_history() -> None:
    prompt = _PromptStub()
    tui = XerxesTUI()
    tui._prompt = prompt  # type: ignore[assignment]

    tui._on_tool_call("tool_1", "AgentTool", '{"prompt":"inspect"}')
    tui._on_tool_result("tool_1", "large completed sub-agent output")

    assert prompt.committed
    assert "Used AgentTool" in prompt.committed[0][1]
    assert "large completed sub-agent output" not in prompt.committed[0][1]

