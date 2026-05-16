from __future__ import annotations

import time

from xerxes.streaming import loop
from xerxes.streaming.events import TextChunk, ToolEnd, ToolStart


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


def test_steer_drain_injects_user_message_between_tool_iterations() -> None:
    """Pending steers must land as a user message before the next LLM call."""
    original = loop._stream_llm
    calls = {"n": 0}
    state = loop.AgentState()
    drained: list[list[str]] = []

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
            yield TextChunk("final")
            yield {"tool_calls": [], "in_tokens": 1, "out_tokens": 1}

    pending = ["please reconsider"]

    def drain():
        out = pending[:]
        pending.clear()
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

    # drain() ran on both iterations.
    assert len(drained) == 2
    assert drained[0] == ["please reconsider"]
    assert drained[1] == []
    # The drained steer landed as a synthetic user message between the
    # tool result and the next LLM call.
    user_messages = [m for m in state.messages if m.get("role") == "user"]
    assert any("please reconsider" in m["content"] for m in user_messages)
    assert any("[mid-turn steer from user]" in m["content"] for m in user_messages)


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
