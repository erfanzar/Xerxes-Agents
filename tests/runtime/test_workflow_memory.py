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

from xerxes.runtime.agent_memory import AgentMemory
from xerxes.runtime.workflow_memory import capture_user_workflow_memory, should_capture_workflow_memory
from xerxes.streaming import loop as streaming_loop
from xerxes.streaming.events import AgentState, TextChunk, TurnDone
from xerxes.tools.agent_memory_tool import active_memory, set_active_memory


def test_workflow_memory_capture_is_generic_for_big_projects(tmp_path):
    previous = active_memory()
    memory = AgentMemory(
        project_root=tmp_path / "repo",
        global_dir=tmp_path / "global",
        project_dir=tmp_path / "project",
    )
    memory.ensure()
    set_active_memory(memory)

    try:
        result = capture_user_workflow_memory(
            "I want this agent to understand big projects and use this workflow across the codebase.",
            project_root=tmp_path / "repo",
        )

        assert result.captured is True
        assert result.scope == "project"
        body = memory.read("project", "WORKFLOW.md")
        assert "understand big projects" in body
        assert "EasyDeL" not in body
    finally:
        set_active_memory(previous)


def test_workflow_memory_does_not_capture_ordinary_project_tasks():
    assert should_capture_workflow_memory("I want this project test suite to pass") is False
    assert should_capture_workflow_memory("please fix this codebase import error") is False


def test_streaming_loop_captures_top_level_workflow_memory(monkeypatch, tmp_path):
    previous = active_memory()
    memory = AgentMemory(
        project_root=tmp_path / "repo",
        global_dir=tmp_path / "global",
        project_dir=tmp_path / "project",
    )
    memory.ensure()
    set_active_memory(memory)

    def fake_stream(*args, **kwargs):
        yield TextChunk("ok")
        yield {
            "tool_calls": [],
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
        }

    monkeypatch.setattr(streaming_loop, "_stream_llm", fake_stream)

    try:
        events = list(
            streaming_loop.run(
                "I want the agent to understand large repos and keep track of my workflow.",
                state=AgentState(),
                config={"model": "gpt-4o", "project_dir": str(tmp_path / "repo")},
                system_prompt="system",
                tool_schemas=[],
            )
        )

        assert any(isinstance(event, TurnDone) for event in events)
        assert "large repos" in memory.read("project", "WORKFLOW.md")
    finally:
        set_active_memory(previous)
