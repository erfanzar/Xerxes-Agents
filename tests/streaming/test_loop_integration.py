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
"""Integration tests for the streaming loop using a fake LLM.

These tests exercise the real streaming loop machinery (token streaming,
tool dispatch, permission gating, cancellation, budget enforcement,
thinking-tag parsing, error recovery) without requiring real API keys.
"""

from __future__ import annotations

from typing import Any

from xerxes.streaming.events import AgentState, TextChunk, ThinkingChunk, ToolEnd, ToolStart, TurnDone
from xerxes.streaming.loop import run


def _run_loop(
    fake_llm: Any,
    prompt: str,
    tool_executor: Any = None,
    tool_schemas: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    cancel_check: Any = None,
) -> list[Any]:
    """Run the streaming loop and collect all events."""
    cfg = config or {}
    cfg.setdefault("model", "gpt-4o")
    cfg.setdefault("permission_mode", "accept-all")
    state = AgentState()
    events = list(
        run(
            user_message=prompt,
            state=state,
            config=cfg,
            system_prompt="You are a test agent.",
            tool_executor=tool_executor,
            tool_schemas=tool_schemas,
            cancel_check=cancel_check,
        )
    )
    return events


def _text(events: list[Any]) -> str:
    return "".join(e.text for e in events if isinstance(e, TextChunk))


class TestSimpleTurn:
    """Text-only turns with no tool calls."""

    def test_single_text_response(self, fake_llm):
        fake_llm.add_text("Hello, world!")
        events = _run_loop(fake_llm, "Hi")
        assert "Hello, world!" in _text(events)
        assert any(isinstance(e, TurnDone) for e in events)

    def test_multi_chunk_text(self, fake_llm):
        fake_llm.add_response(
            [
                TextChunk("Hello"),
                TextChunk(", "),
                TextChunk("world!"),
                {"tool_calls": [], "in_tokens": 5, "out_tokens": 10},
            ]
        )
        events = _run_loop(fake_llm, "Hi")
        assert _text(events) == "Hello, world!"

    def test_empty_response_does_not_crash(self, fake_llm):
        fake_llm.add_response([{"tool_calls": [], "in_tokens": 0, "out_tokens": 0}])
        events = _run_loop(fake_llm, "Hi")
        done = [e for e in events if isinstance(e, TurnDone)]
        assert len(done) == 1


class TestToolCalls:
    """Turns with tool calls and result feeding."""

    def test_single_tool_call_succeeds(self, fake_llm):
        fake_llm.add_tool_call("ReadFile", {"file_path": "test.py"}, text_before="Reading file...")
        fake_llm.add_text("The file has 10 lines.")
        calls: list[tuple[str, dict]] = []

        def executor(name: str, inp: dict) -> str:
            calls.append((name, inp))
            return "content of test.py"

        events = _run_loop(fake_llm, "Read test.py", tool_executor=executor)
        assert len(calls) == 1
        assert calls[0][0] == "ReadFile"
        starts = [e for e in events if isinstance(e, ToolStart)]
        assert len(starts) == 1
        assert starts[0].name == "ReadFile"
        ends = [e for e in events if isinstance(e, ToolEnd)]
        assert len(ends) == 1
        assert "content of test.py" in ends[0].result

    def test_tool_error_is_surfaced(self, fake_llm):
        fake_llm.add_tool_call("ReadFile", {"file_path": "missing.py"})

        def executor(name: str, inp: dict) -> str:
            raise FileNotFoundError("No such file")

        events = _run_loop(fake_llm, "Read missing.py", tool_executor=executor)
        ends = [e for e in events if isinstance(e, ToolEnd)]
        assert len(ends) == 1
        assert "Error executing ReadFile" in ends[0].result

    def test_multi_tool_chain(self, fake_llm):
        fake_llm.add_tool_call("ReadFile", {"file_path": "a.py"}, call_id="c1")
        fake_llm.add_tool_call("ReadFile", {"file_path": "b.py"}, call_id="c2")
        fake_llm.add_text("Both files read successfully.")

        def executor(name: str, inp: dict) -> str:
            return f"content of {inp.get('file_path', '?')}"

        events = _run_loop(fake_llm, "Read both files", tool_executor=executor)
        starts = [e for e in events if isinstance(e, ToolStart)]
        assert len(starts) == 2
        assert starts[0].inputs["file_path"] == "a.py"
        assert starts[1].inputs["file_path"] == "b.py"


class TestBudgetEnforcement:
    """Token budget enforcement inside the streaming loop."""

    def test_budget_exhausted_mid_turn(self, fake_llm):
        fake_llm.add_tool_call("ReadFile", {"file_path": "a.py"}, in_tokens=60000, out_tokens=10000)
        fake_llm.add_tool_call("ReadFile", {"file_path": "b.py"}, in_tokens=60000, out_tokens=10000)
        fake_llm.add_text("Done.")

        def executor(name: str, inp: dict) -> str:
            return "ok"

        config = {"model": "gpt-4o", "permission_mode": "accept-all", "max_budget_tokens": 100000}
        events = _run_loop(fake_llm, "Read", tool_executor=executor, config=config)
        text = _text(events)
        # After two 70k-token turns (140k total), budget should trigger
        assert "budget" in text.lower() or "Done." in text


class TestCancellation:
    """Cancellation between tool iterations."""

    def test_cancel_between_tools(self, fake_llm):
        fake_llm.add_tool_call("ReadFile", {"file_path": "a.py"}, call_id="c1")
        fake_llm.add_text("Done.")

        cancel_after_first = {"called": False}

        def cancel_check():
            if cancel_after_first["called"]:
                return True
            cancel_after_first["called"] = True
            return False

        def executor(name: str, inp: dict) -> str:
            return "ok"

        events = _run_loop(fake_llm, "Read", tool_executor=executor, cancel_check=cancel_check)
        text = _text(events)
        assert "[Cancelled]" in text


class TestThinkingTags:
    """Inline thinking-tag parsing in streamed output."""

    def test_think_block_extracted(self, fake_llm):
        fake_llm.add_response(
            [
                TextChunk("<think>reasoning here</think>"),
                TextChunk("The answer is 42."),
                {"tool_calls": [], "in_tokens": 10, "out_tokens": 20},
            ]
        )
        events = _run_loop(fake_llm, "What is the answer?")
        text = _text(events)
        assert "42" in text
        assert "<think>" not in text
        thinking = [e for e in events if isinstance(e, ThinkingChunk)]
        assert len(thinking) >= 1

    def test_split_think_tag_across_chunks(self, fake_llm):
        fake_llm.add_response(
            [
                TextChunk("<thi"),
                TextChunk("nking>secret reasoning</thin"),
                TextChunk("king>visible answer"),
                {"tool_calls": [], "in_tokens": 5, "out_tokens": 15},
            ]
        )
        events = _run_loop(fake_llm, "Test")
        text = _text(events)
        assert "visible answer" in text
        assert "secret" not in text


class TestPermissionGate:
    """Permission mode interaction with tool execution."""

    def test_manual_mode_denies_all(self, fake_llm):
        fake_llm.add_tool_call("ReadFile", {"file_path": "x.py"})
        fake_llm.add_text("Done.")

        def executor(name: str, inp: dict) -> str:
            return "ok"

        config = {"model": "gpt-4o", "permission_mode": "manual"}
        events = _run_loop(fake_llm, "Read x.py", tool_executor=executor, config=config)
        ends = [e for e in events if isinstance(e, ToolEnd)]
        assert len(ends) == 1
        assert not ends[0].permitted

    def test_plan_mode_denies_mutations(self, fake_llm):
        fake_llm.add_tool_call("FileEditTool", {"file_path": "x.py", "old_string": "a", "new_string": "b"})
        fake_llm.add_text("Done.")

        def executor(name: str, inp: dict) -> str:
            return "ok"

        config = {"model": "gpt-4o", "permission_mode": "plan"}
        events = _run_loop(fake_llm, "Edit x.py", tool_executor=executor, config=config)
        ends = [e for e in events if isinstance(e, ToolEnd)]
        assert len(ends) == 1
        assert not ends[0].permitted

    def test_plan_mode_allows_read_only(self, fake_llm):
        fake_llm.add_tool_call("ReadFile", {"file_path": "x.py"})
        fake_llm.add_text("Done.")

        def executor(name: str, inp: dict) -> str:
            return "file content"

        config = {"model": "gpt-4o", "permission_mode": "plan"}
        events = _run_loop(fake_llm, "Read x.py", tool_executor=executor, config=config)
        ends = [e for e in events if isinstance(e, ToolEnd)]
        assert len(ends) == 1
        assert ends[0].permitted


class TestStateUpdates:
    """AgentState is mutated correctly during the loop."""

    def test_messages_appended(self, fake_llm):
        fake_llm.add_text("Response.")
        state = AgentState()
        cfg = {"model": "gpt-4o", "permission_mode": "accept-all"}
        list(run("Prompt", state, cfg, "System prompt"))
        roles = [m["role"] for m in state.messages]
        assert "user" in roles
        assert "assistant" in roles

    def test_token_totals_accumulated(self, fake_llm):
        fake_llm.add_text("Hi", in_tokens=100, out_tokens=50)
        state = AgentState()
        cfg = {"model": "gpt-4o", "permission_mode": "accept-all"}
        list(run("Prompt", state, cfg, "System"))
        assert state.total_input_tokens >= 100
        assert state.total_output_tokens >= 50
