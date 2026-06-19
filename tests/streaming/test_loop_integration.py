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
from xerxes.streaming.loop import _try_compact_messages, run


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

    @staticmethod
    def _summary_agent(messages: list[dict[str, Any]], _previous_summary: str | None) -> str:
        return "agent summary"

    def _compaction_config(self) -> dict[str, Any]:
        return {
            "model": "gpt-4o",
            "compaction_summary_agent": self._summary_agent,
            "compaction_target_tokens": 80,
        }

    def _assert_valid_tool_sequence(self, messages: list[dict]) -> None:
        pending: set[str] = set()
        for message in messages:
            role = message.get("role")
            if role == "assistant":
                assert not pending
                pending = {tc["id"] for tc in message.get("tool_calls") or []}
            elif role == "tool":
                assert message["tool_call_id"] in pending
                pending.remove(message["tool_call_id"])
            else:
                assert not pending
        assert not pending

    def test_budget_exhausted_mid_turn(self, fake_llm):
        fake_llm.add_tool_call("ReadFile", {"file_path": "a.py"}, in_tokens=110000, out_tokens=10000)
        fake_llm.add_text("Done.")

        def executor(name: str, inp: dict) -> str:
            return "ok"

        config = {"model": "gpt-4o", "permission_mode": "accept-all", "max_budget_tokens": 100000}
        events = _run_loop(fake_llm, "Read", tool_executor=executor, config=config)
        text = _text(events)
        assert "session token budget" in text.lower()
        assert "Done." not in text
        assert fake_llm.call_count == 1

    def test_default_session_token_budget_is_uncapped(self, fake_llm):
        fake_llm.add_tool_call("ReadFile", {"file_path": "a.py"}, in_tokens=540000, out_tokens=20000)
        fake_llm.add_text("Done.")

        def executor(name: str, inp: dict) -> str:
            return "ok"

        events = _run_loop(fake_llm, "Read", tool_executor=executor)
        text = _text(events)
        assert "session token budget" not in text.lower()
        assert "Done." in text
        assert fake_llm.call_count == 2

    def test_budget_compaction_keeps_tool_call_parent_for_tail_tool_results(self):
        tool_calls = [{"id": f"call_{idx}", "name": "TaskGetTool", "input": {"id": str(idx)}} for idx in range(8)]
        state = AgentState(
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "scan repository " * 200},
                {"role": "assistant", "content": "starting " * 200},
                {"role": "assistant", "content": "", "tool_calls": tool_calls},
                *[
                    {
                        "role": "tool",
                        "tool_call_id": f"call_{idx}",
                        "name": "TaskGetTool",
                        "content": f"result {idx}",
                    }
                    for idx in range(8)
                ],
            ],
            total_input_tokens=100_000,
            total_output_tokens=100_000,
        )

        assert _try_compact_messages(state, budget_limit=100_000, config=self._compaction_config()) is True

        seen_tool_call_ids: set[str] = set()
        for message in state.messages:
            if message.get("role") == "assistant":
                seen_tool_call_ids.update(tc["id"] for tc in message.get("tool_calls") or [])
            if message.get("role") == "tool":
                assert message["tool_call_id"] in seen_tool_call_ids

        assert state.messages[-9]["role"] == "assistant"
        assert len(state.messages[-9]["tool_calls"]) == 8
        self._assert_valid_tool_sequence(state.messages)

    def test_budget_compaction_drops_orphan_tool_results(self):
        state = AgentState(
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "old request " * 200},
                {"role": "assistant", "content": "old answer " * 200},
                {"role": "user", "content": "recent request"},
                {"role": "assistant", "content": "recent answer"},
                {"role": "user", "content": "tail request"},
                {
                    "role": "tool",
                    "tool_call_id": "missing-parent",
                    "name": "TaskOutputTool",
                    "content": "orphan result",
                },
            ],
            total_input_tokens=100_000,
            total_output_tokens=100_000,
        )

        assert _try_compact_messages(state, budget_limit=100_000, config=self._compaction_config()) is True

        assert all(message.get("tool_call_id") != "missing-parent" for message in state.messages)
        self._assert_valid_tool_sequence(state.messages)

    def test_budget_compaction_backfills_missing_recent_tool_result(self):
        state = AgentState(
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "old request " * 200},
                {"role": "assistant", "content": "old answer " * 200},
                {"role": "user", "content": "recent request"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call_missing", "name": "TaskOutputTool", "input": {"task_id": "t1"}}],
                },
                {"role": "user", "content": "next request"},
            ],
            total_input_tokens=100_000,
            total_output_tokens=100_000,
        )

        assert _try_compact_messages(state, budget_limit=100_000, config=self._compaction_config()) is True

        tool_messages = [message for message in state.messages if message.get("role") == "tool"]
        assert any(message.get("tool_call_id") == "call_missing" for message in tool_messages)
        self._assert_valid_tool_sequence(state.messages)

    def test_context_compacts_after_tool_results_before_next_request(self, fake_llm):
        fake_llm.add_tool_call("ReadFile", {"file_path": "a.py"}, call_id="c1")
        fake_llm.add_text("Done.")
        state = AgentState(
            messages=[
                {"role": "user", "content": "old " * 100},
                {"role": "assistant", "content": "prior " * 50},
            ]
        )
        config = {
            "model": "gpt-4o",
            "permission_mode": "accept-all",
            "max_context_tokens": 500,
            "compaction_threshold_tokens": 300,
            "compaction_target_tokens": 120,
            "compaction_summary_agent": self._summary_agent,
        }

        events = list(
            run(
                user_message="Read",
                state=state,
                config=config,
                system_prompt="You are a test agent.",
                tool_executor=lambda _name, _inp: "data " * 300,
                tool_schemas=[],
            )
        )

        text = _text(events)
        assert "Context compacted after tool results" in text
        assert "Done." in text
        assert state.metadata["last_compaction"]["tokens_before"] > state.metadata["last_compaction"]["tokens_after"]
        self._assert_valid_tool_sequence(state.messages)

    def test_context_stops_after_tool_results_when_compaction_unavailable(self, fake_llm):
        fake_llm.add_tool_call("ReadFile", {"file_path": "a.py"}, call_id="c1")
        fake_llm.add_text("Done.")
        state = AgentState(
            messages=[
                {"role": "user", "content": "old " * 100},
                {"role": "assistant", "content": "prior " * 50},
            ]
        )
        config = {
            "model": "gpt-4o",
            "permission_mode": "accept-all",
            "max_context_tokens": 300,
            "compaction_threshold_tokens": 200,
            "compaction_summary_agent": False,
        }

        events = list(
            run(
                user_message="Read",
                state=state,
                config=config,
                system_prompt="You are a test agent.",
                tool_executor=lambda _name, _inp: "data " * 300,
                tool_schemas=[],
            )
        )

        text = _text(events)
        assert "exceeded after tool results" in text
        assert "Done." not in text
        assert fake_llm.call_count == 1
        self._assert_valid_tool_sequence(state.messages)


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
