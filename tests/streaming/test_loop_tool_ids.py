from __future__ import annotations

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

