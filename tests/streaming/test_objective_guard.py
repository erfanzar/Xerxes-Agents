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
"""Objective mode is enforced by the stream loop, not only by prompt text."""

from __future__ import annotations

from types import SimpleNamespace

from xerxes.runtime.config_context import get_active_config
from xerxes.runtime.session_context import set_active_session
from xerxes.streaming import loop
from xerxes.streaming.events import TextChunk
from xerxes.tools.claude_tools import SetInteractionModeTool


def test_objective_mode_continues_after_unmet_no_tool_answer(fake_llm) -> None:
    state = loop.AgentState()
    fake_llm.add_text("Here's the honest final state: ❌ still losing. Want me to port MLX?")
    fake_llm.add_text("Verified complete: all benchmarks pass and all acceptance criteria pass.")

    events = list(
        loop.run(
            user_message="beat jax-mps on every matmul benchmark",
            state=state,
            config={"model": "openai/test", "permission_mode": "accept-all", "mode": "objective"},
            system_prompt="",
            tool_executor=lambda name, inp: "ok",
            tool_schemas=[],
        )
    )

    text = "".join(event.text for event in events if isinstance(event, TextChunk))
    guard_messages = [message for message in state.messages if message.get("content", "").startswith("[Objective gate]")]

    assert fake_llm.call_count == 2
    assert guard_messages
    assert "Objective gate: unresolved acceptance marker" in text
    assert "Verified complete" in text


def test_code_mode_allows_same_no_tool_answer(fake_llm) -> None:
    state = loop.AgentState()
    fake_llm.add_text("Here's the honest final state: ❌ still losing. Want me to port MLX?")

    events = list(
        loop.run(
            user_message="inspect benchmark results",
            state=state,
            config={"model": "openai/test", "permission_mode": "accept-all", "mode": "code"},
            system_prompt="",
            tool_executor=lambda name, inp: "ok",
            tool_schemas=[],
        )
    )

    text = "".join(event.text for event in events if isinstance(event, TextChunk))

    assert fake_llm.call_count == 1
    assert "Objective gate:" not in text


def test_model_mode_switch_to_code_disarms_objective_guard_in_same_turn(fake_llm) -> None:
    state = loop.AgentState()
    session = SimpleNamespace(
        interaction_mode="objective",
        plan_mode=False,
        runtime_config={"mode": "objective", "plan_mode": False},
        key="tui:test",
        id="test",
    )
    fake_llm.add_tool_call("SetInteractionModeTool", {"mode": "code", "reason": "Verified enough to report"})
    fake_llm.add_text("Here's the honest final state: ❌ still losing. Want me to port MLX?")

    def executor(name: str, inputs: dict) -> str:
        assert name == "SetInteractionModeTool"
        return SetInteractionModeTool.static_call(**inputs)

    set_active_session(session)
    try:
        events = list(
            loop.run(
                user_message="beat jax-mps on every matmul benchmark",
                state=state,
                config={"model": "openai/test", "permission_mode": "accept-all", "mode": "objective"},
                system_prompt="",
                tool_executor=executor,
                tool_schemas=[],
            )
        )
    finally:
        set_active_session(None)

    text = "".join(event.text for event in events if isinstance(event, TextChunk))

    assert get_active_config() is None
    assert fake_llm.call_count == 2
    assert session.interaction_mode == "code"
    assert session.runtime_config["mode"] == "code"
    assert "Objective gate:" not in text
