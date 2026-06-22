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

import json
import time

from xerxes.runtime.agent_memory import AgentMemory
from xerxes.streaming import loop
from xerxes.streaming.events import TextChunk, ToolEnd, ToolStart, TurnDone
from xerxes.tools.agent_memory_tool import active_memory, set_active_memory


def test_run_assigns_tool_id_when_provider_omits_one() -> None:
    original = loop._stream_llm
    calls = {"n": 0}

    def fake_stream(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            yield {
                "tool_calls": [
                    {
                        "id": "",
                        "name": "ExecuteShell",
                        "input": {"command": "cd /tmp && pwd"},
                    }
                ],
                "in_tokens": 1,
                "out_tokens": 1,
            }
        else:
            yield TextChunk("done")
            yield {"tool_calls": [], "in_tokens": 1, "out_tokens": 1}

    loop._stream_llm = fake_stream
    try:
        events = list(
            loop.run(
                user_message="test",
                state=loop.AgentState(),
                config={"model": "openai/test", "permission_mode": "accept-all"},
                system_prompt="",
                tool_executor=lambda name, inp: "shell result",
                tool_schemas=[],
            )
        )
    finally:
        loop._stream_llm = original

    starts = [event for event in events if isinstance(event, ToolStart)]
    ends = [event for event in events if isinstance(event, ToolEnd)]

    assert len(starts) == 1
    assert len(ends) == 1
    assert starts[0].tool_call_id
    assert ends[0].tool_call_id == starts[0].tool_call_id


def test_run_has_no_hidden_fifty_tool_turn_cap() -> None:
    original = loop._stream_llm
    calls = {"n": 0}

    def fake_stream(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 55:
            yield {
                "tool_calls": [
                    {
                        "id": f"call_{calls['n']}",
                        "name": "ReadFile",
                        "input": {"file_path": "x"},
                    }
                ],
                "in_tokens": 1,
                "out_tokens": 1,
            }
        else:
            yield TextChunk("done")
            yield {"tool_calls": [], "in_tokens": 1, "out_tokens": 1}

    loop._stream_llm = fake_stream
    try:
        events = list(
            loop.run(
                user_message="test",
                state=loop.AgentState(),
                config={"model": "openai/test", "permission_mode": "accept-all"},
                system_prompt="",
                tool_executor=lambda name, inp: "ok",
                tool_schemas=[],
            )
        )
    finally:
        loop._stream_llm = original

    text = "".join(event.text for event in events if isinstance(event, TextChunk))
    ends = [event for event in events if isinstance(event, ToolEnd)]

    assert calls["n"] == 56
    assert len(ends) == 55
    assert "reached max tool turns (50)" not in text
    assert "done" in text


def test_run_honors_configured_max_tool_turns() -> None:
    original = loop._stream_llm
    calls = {"n": 0}

    def fake_stream(*args, **kwargs):
        calls["n"] += 1
        yield {
            "tool_calls": [
                {
                    "id": f"call_{calls['n']}",
                    "name": "ReadFile",
                    "input": {"file_path": "x"},
                }
            ],
            "in_tokens": 1,
            "out_tokens": 1,
        }

    loop._stream_llm = fake_stream
    try:
        events = list(
            loop.run(
                user_message="test",
                state=loop.AgentState(),
                config={
                    "model": "openai/test",
                    "permission_mode": "accept-all",
                    "max_tool_turns": 2,
                },
                system_prompt="",
                tool_executor=lambda name, inp: "ok",
                tool_schemas=[],
            )
        )
    finally:
        loop._stream_llm = original

    text = "".join(event.text for event in events if isinstance(event, TextChunk))
    ends = [event for event in events if isinstance(event, ToolEnd)]

    assert calls["n"] == 2
    assert len(ends) == 2
    assert "reached configured max tool turns (2)" in text


def test_run_preserves_full_tool_result_for_model_context() -> None:
    original = loop._stream_llm
    long_result = "x" * 5000
    calls = {"n": 0}
    state = loop.AgentState()

    def fake_stream(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            yield {
                "tool_calls": [
                    {
                        "id": "call_read",
                        "name": "ReadFile",
                        "input": {"file_path": "big.txt"},
                    }
                ],
                "in_tokens": 1,
                "out_tokens": 1,
            }
        else:
            yield TextChunk("done")
            yield {"tool_calls": [], "in_tokens": 1, "out_tokens": 1}

    loop._stream_llm = fake_stream
    try:
        events = list(
            loop.run(
                user_message="test",
                state=state,
                config={"model": "openai/test", "permission_mode": "accept-all"},
                system_prompt="",
                tool_executor=lambda name, inp: long_result,
                tool_schemas=[],
            )
        )
    finally:
        loop._stream_llm = original

    ends = [event for event in events if isinstance(event, ToolEnd)]
    tool_messages = [msg for msg in state.messages if msg.get("role") == "tool"]

    assert ends[0].result == long_result
    assert tool_messages[0]["content"] == long_result


def test_run_spills_large_tool_result_to_project_memory(tmp_path) -> None:
    original = loop._stream_llm
    previous_memory = active_memory()
    memory = AgentMemory(project_root=tmp_path)
    large_result = "FULL-RESULT-LINE\n" * 500
    calls = {"n": 0}
    seen_provider_messages: list[list[dict]] = []
    summary_inputs: dict[str, str] = {}
    state = loop.AgentState()

    def summary_agent(messages, _previous_summary):
        summary_inputs["content"] = messages[0]["content"]
        return "compact summary with the important facts"

    def fake_stream(*args, **kwargs):
        calls["n"] += 1
        seen_provider_messages.append(list(kwargs["messages"]))
        if calls["n"] == 1:
            yield {
                "tool_calls": [
                    {
                        "id": "call_large",
                        "name": "ExecuteShell",
                        "input": {"command": "make noisy"},
                    }
                ],
                "in_tokens": 1,
                "out_tokens": 1,
            }
        else:
            yield TextChunk("done")
            yield {"tool_calls": [], "in_tokens": 1, "out_tokens": 1}

    loop._stream_llm = fake_stream
    set_active_memory(memory)
    try:
        events = list(
            loop.run(
                user_message="run noisy command",
                state=state,
                config={
                    "model": "openai/test",
                    "permission_mode": "accept-all",
                    "tool_result_spill_chars": 100,
                    "compaction_summary_agent": summary_agent,
                },
                system_prompt="",
                tool_executor=lambda name, inp: large_result,
                tool_schemas=[],
            )
        )
    finally:
        set_active_memory(previous_memory)
        loop._stream_llm = original

    ends = [event for event in events if isinstance(event, ToolEnd)]
    tool_messages = [msg for msg in state.messages if msg.get("role") == "tool"]
    files = [entry for entry in memory.list_files("project") if entry.relative.startswith("tool-results/")]

    assert len(files) == 1
    assert memory.read("project", files[0].relative) == large_result
    assert summary_inputs["content"] == large_result
    assert "[Large tool result stored outside model context]" in ends[0].result
    assert "agent_memory_read" in ends[0].result
    assert "compact summary with the important facts" in ends[0].result
    assert large_result not in ends[0].result
    assert tool_messages[0]["content"] == ends[0].result
    assert large_result not in seen_provider_messages[1][-1]["content"]


def test_large_tool_result_when_memory_init_fails_returns_pointer_error_not_raw_payload(tmp_path, monkeypatch) -> None:
    original = loop._stream_llm
    previous_memory = active_memory()
    large_result = "NOISY-RAW-OUTPUT\n" * 500
    calls = {"n": 0}
    seen_provider_messages: list[list[dict]] = []
    state = loop.AgentState()

    def fake_stream(*args, **kwargs):
        calls["n"] += 1
        seen_provider_messages.append(list(kwargs["messages"]))
        if calls["n"] == 1:
            yield {
                "tool_calls": [
                    {
                        "id": "call_large",
                        "name": "ExecuteShell",
                        "input": {"command": "make noisy"},
                    }
                ],
                "in_tokens": 1,
                "out_tokens": 1,
            }
        else:
            yield TextChunk("done")
            yield {"tool_calls": [], "in_tokens": 1, "out_tokens": 1}

    from xerxes.runtime.agent_memory import AgentMemory

    def fail_ensure(self):
        raise OSError("readonly memory root")

    monkeypatch.setattr(AgentMemory, "ensure", fail_ensure)
    loop._stream_llm = fake_stream
    set_active_memory(None)
    try:
        events = list(
            loop.run(
                user_message="run noisy command",
                state=state,
                config={
                    "model": "openai/test",
                    "permission_mode": "accept-all",
                    "project_dir": str(tmp_path / "repo"),
                    "tool_result_spill_chars": 100,
                    "compaction_summary_agent": False,
                },
                system_prompt="",
                tool_executor=lambda name, inp: large_result,
                tool_schemas=[],
            )
        )
    finally:
        set_active_memory(previous_memory)
        loop._stream_llm = original

    ends = [event for event in events if isinstance(event, ToolEnd)]
    tool_messages = [msg for msg in state.messages if msg.get("role") == "tool"]

    assert "[Large tool result stored outside model context]" in ends[0].result
    assert "project agent memory could not initialize" in ends[0].result
    assert "readonly memory root" in ends[0].result
    assert large_result not in ends[0].result
    assert tool_messages[0]["content"] == ends[0].result
    assert large_result not in seen_provider_messages[1][-1]["content"]


def test_steer_drain_injects_user_message_between_tool_iterations() -> None:
    """Pending steers must land as a user message before the next LLM call."""
    original = loop._stream_llm
    calls = {"n": 0}
    state = loop.AgentState()
    drained: list[list[str]] = []
    provider_messages: list[list[dict]] = []

    def fake_stream(*args, **kwargs):
        calls["n"] += 1
        provider_messages.append([dict(msg) for msg in kwargs["messages"]])
        if calls["n"] == 1:
            yield {
                "tool_calls": [
                    {"id": "c1", "name": "Bash", "input": {"command": "ls"}},
                ],
                "in_tokens": 1,
                "out_tokens": 1,
            }
        else:
            yield TextChunk("final")
            yield {"tool_calls": [], "in_tokens": 1, "out_tokens": 1}

    pending_batches = [[], ["please reconsider"], []]

    def drain():
        out = pending_batches.pop(0) if pending_batches else []
        drained.append(out)
        return out

    loop._stream_llm = fake_stream
    try:
        list(
            loop.run(
                user_message="initial",
                state=state,
                config={"model": "openai/test", "permission_mode": "accept-all"},
                system_prompt="",
                tool_executor=lambda name, inp: "ok",
                tool_schemas=[],
                steer_drain=drain,
            )
        )
    finally:
        loop._stream_llm = original

    assert drained == [[], ["please reconsider"], []]
    # The drained steer landed as a synthetic user message between the
    # tool result and the next LLM call.
    user_messages = [m for m in state.messages if m.get("role") == "user"]
    assert any("please reconsider" in m["content"] for m in user_messages)
    assert any("[mid-turn steer from user]" in m["content"] for m in user_messages)
    assert "please reconsider" not in json.dumps(provider_messages[0])
    assert "please reconsider" in json.dumps(provider_messages[1])


def test_late_steer_after_no_tool_response_is_saved_for_next_turn() -> None:
    """A steer queued during a final no-tool response must not linger stale."""
    original = loop._stream_llm
    state = loop.AgentState()
    drained: list[list[str]] = []
    pending_batches = [[], ["make a todo for it"]]

    def fake_stream(*args, **kwargs):
        yield TextChunk("done")
        yield {"tool_calls": [], "in_tokens": 1, "out_tokens": 1}

    def drain():
        out = pending_batches.pop(0) if pending_batches else []
        drained.append(out)
        return out

    loop._stream_llm = fake_stream
    try:
        events = list(
            loop.run(
                user_message="finish this",
                state=state,
                config={"model": "openai/test", "permission_mode": "accept-all"},
                system_prompt="",
                tool_executor=lambda name, inp: "",
                tool_schemas=[],
                steer_drain=drain,
            )
        )
    finally:
        loop._stream_llm = original

    assert drained == [[], ["make a todo for it"]]
    assert any(
        isinstance(event, TextChunk) and "Steer saved for next turn: make a todo for it" in event.text
        for event in events
    )
    user_messages = [m for m in state.messages if m.get("role") == "user"]
    assert any("[steer from user saved for next turn]" in m["content"] for m in user_messages)
    assert any("make a todo for it" in m["content"] for m in user_messages)


def test_agent_event_drain_injects_synthetic_user_message() -> None:
    """Drained agent-event lines should land as a [sub-agent events] message."""
    original = loop._stream_llm
    calls = {"n": 0}
    state = loop.AgentState()

    def fake_stream(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            yield {
                "tool_calls": [
                    {"id": "c1", "name": "Bash", "input": {"command": "ls"}},
                ],
                "in_tokens": 1,
                "out_tokens": 1,
            }
        else:
            yield TextChunk("ack")
            yield {"tool_calls": [], "in_tokens": 1, "out_tokens": 1}

    pending = ["[agent worker] spawned", "[agent worker] → Read(file=x)"]

    def drain():
        out = pending[:]
        pending.clear()
        return out

    loop._stream_llm = fake_stream
    try:
        list(
            loop.run(
                user_message="please track",
                state=state,
                config={"model": "openai/test", "permission_mode": "accept-all"},
                system_prompt="",
                tool_executor=lambda name, inp: "ok",
                tool_schemas=[],
                agent_event_drain=drain,
            )
        )
    finally:
        loop._stream_llm = original

    user_messages = [m for m in state.messages if m.get("role") == "user"]
    assert any("[sub-agent events]" in m["content"] for m in user_messages)
    assert any("[agent worker] spawned" in m["content"] for m in user_messages)


def test_steer_drain_noop_when_empty() -> None:
    """An empty drain must not append spurious user messages."""
    original = loop._stream_llm
    state = loop.AgentState()

    def fake_stream(*args, **kwargs):
        yield TextChunk("done")
        yield {"tool_calls": [], "in_tokens": 1, "out_tokens": 1}

    loop._stream_llm = fake_stream
    try:
        list(
            loop.run(
                user_message="ping",
                state=state,
                config={"model": "openai/test", "permission_mode": "accept-all"},
                system_prompt="",
                tool_executor=lambda name, inp: "",
                tool_schemas=[],
                steer_drain=lambda: [],
            )
        )
    finally:
        loop._stream_llm = original

    user_messages = [m for m in state.messages if m.get("role") == "user"]
    assert len(user_messages) == 1
    assert user_messages[0]["content"] == "ping"


def test_llm_stream_retry_uses_six_fixed_delay_stages(monkeypatch) -> None:
    original = loop._stream_llm
    calls = {"n": 0}
    sleeps: list[int] = []

    def fake_stream(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 6:
            raise ConnectionError("Connection error.")
        yield TextChunk("done")
        yield {"tool_calls": [], "in_tokens": 1, "out_tokens": 1}

    loop._stream_llm = fake_stream
    monkeypatch.setattr(loop.time, "sleep", sleeps.append)
    try:
        events = list(
            loop.run(
                user_message="test",
                state=loop.AgentState(),
                config={"model": "openai/test", "permission_mode": "accept-all"},
                system_prompt="",
                tool_executor=lambda name, inp: "ok",
                tool_schemas=[],
            )
        )
    finally:
        loop._stream_llm = original

    text = "".join(event.text for event in events if isinstance(event, TextChunk))

    assert calls["n"] == 7
    assert sleeps == [5, 5, 5, 5, 5, 5]
    assert "Retrying in 5s... (6/6)" in text
    assert "done" in text
    assert any(isinstance(event, TurnDone) for event in events)


def test_tool_end_duration_reflects_executor_time() -> None:
    original = loop._stream_llm
    calls = {"n": 0}

    def fake_stream(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            yield {
                "tool_calls": [
                    {
                        "id": "call_sleep",
                        "name": "ExecuteShell",
                        "input": {"command": "sleep 0.05"},
                    }
                ],
                "in_tokens": 1,
                "out_tokens": 1,
            }
        else:
            yield TextChunk("done")
            yield {"tool_calls": [], "in_tokens": 1, "out_tokens": 1}

    def slow_executor(name, inp):
        time.sleep(0.05)
        return "ok"

    loop._stream_llm = fake_stream
    try:
        events = list(
            loop.run(
                user_message="test",
                state=loop.AgentState(),
                config={"model": "openai/test", "permission_mode": "accept-all"},
                system_prompt="",
                tool_executor=slow_executor,
                tool_schemas=[],
            )
        )
    finally:
        loop._stream_llm = original

    ends = [event for event in events if isinstance(event, ToolEnd)]
    assert ends[0].duration_ms >= 40
