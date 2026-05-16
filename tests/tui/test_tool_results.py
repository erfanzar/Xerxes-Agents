from __future__ import annotations

from xerxes.tui.app import XerxesTUI
from xerxes.tui.blocks import _ToolCallBlock


class _PromptStub:
    def __init__(self) -> None:
        self.committed: list[tuple[str, str]] = []

    def commit_streaming(self) -> None:
        pass

    def set_spinner_label(self, _label: str) -> None:
        pass

    def set_plan_mode(self, _enabled: bool) -> None:
        pass

    def set_activity_mode(self, _mode: str) -> None:
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
    tui._on_tool_result("tool_1", "large completed sub-agent output", duration_ms=55.0)

    assert prompt.committed
    assert "Used AgentTool" in prompt.committed[0][1]
    assert "55ms" in prompt.committed[0][1]
    assert "large completed sub-agent output" not in prompt.committed[0][1]


def test_tool_block_renders_last_five_subagent_tool_calls() -> None:
    block = _ToolCallBlock("block", "parent", "SpawnAgents", '{"agents":[]}')

    for idx in range(12):
        block.append_sub_tool_call(f"sub-{idx}", f"Tool-{idx:02d}", "{}")

    rendered = block.compose()

    for idx in range(7):
        assert f"Tool-{idx:02d}" not in rendered
    for idx in range(7, 12):
        assert f"Tool-{idx:02d}" in rendered


def test_tool_block_does_not_leak_markup_tags_for_json_list_args() -> None:
    block = _ToolCallBlock(
        "block",
        "parent",
        "SpawnAgents",
        '{"agents":[{"prompt":"Analyze project structure. Scan all directories up to 4 levels deep"}]}',
    )

    rendered = block.compose()

    assert "[/dim]" not in rendered
    assert "[dim]" not in rendered
    assert "Used" not in rendered
    assert "Using SpawnAgents" in rendered
    assert "({'prompt'" in rendered
