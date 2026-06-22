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

import asyncio
from types import SimpleNamespace

from xerxes.tui.app import XerxesTUI
from xerxes.tui.blocks import _ToolCallBlock


class _PromptStub:
    def __init__(self) -> None:
        self.committed: list[tuple[str, str]] = []
        self.skills: list[str] | None = None

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

    def set_session(self, *, agent_name: str, model: str, cwd: str, branch: str = "") -> None:
        pass

    def set_context(self, _current: int, _limit: int) -> None:
        pass

    def set_skills(self, skills: list[str]) -> None:
        self.skills = skills


def test_agent_tool_result_is_not_rendered_in_tui_history() -> None:
    prompt = _PromptStub()
    tui = XerxesTUI()
    tui._prompt = prompt  # type: ignore[assignment]

    tui._on_tool_call("tool_1", "AgentTool", '{"prompt":"inspect"}')
    tui._on_tool_result("tool_1", "large completed sub-agent output", duration_ms=55.0)

    assert prompt.committed
    assert "AgentTool" in prompt.committed[0][1]
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
    assert "SpawnAgents" in rendered
    assert "({'prompt'" in rendered


def test_init_done_clears_skill_completions_when_daemon_sends_empty_list() -> None:
    prompt = _PromptStub()
    prompt.skills = ["stale-skill"]
    tui = XerxesTUI()
    tui._prompt = prompt  # type: ignore[assignment]

    async def fake_load_models() -> list[str]:
        return []

    async def drive() -> None:
        tui._load_models = fake_load_models  # type: ignore[method-assign]
        tui._on_init_done(
            SimpleNamespace(
                session_id="s",
                model="m",
                cwd="/tmp",
                agent_name="default",
                git_branch="",
                context_limit=0,
                skills=[],
            )
        )
        await asyncio.sleep(0)

    asyncio.run(drive())

    assert prompt.skills == []
