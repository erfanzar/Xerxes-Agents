# Copyright 2026 Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
"""Message-layer hardening: tool-result bounding, is_error flag, thinking replay.

These guard the streaming-loop self-correction + context-safety fixes:
  * ``bound_tool_result`` clamps an oversized tool output so one result can't
    poison the window (kept head+tail, idempotent).
  * Anthropic tool_result blocks carry ``is_error`` so the model recognises
    failures instead of treating them as success.
  * Anthropic thinking blocks are only replayed WITH their signature (Anthropic
    400s otherwise), and round-trip cleanly back to the neutral format.
"""

from __future__ import annotations

from xerxes.streaming.messages import (
    MAX_TOOL_RESULT_CHARS,
    bound_tool_result,
    messages_from_anthropic,
    messages_to_anthropic,
)


def test_bound_tool_result_passthrough_small():
    assert bound_tool_result("hello") == "hello"


def test_bound_tool_result_clamps_large_keeping_head_and_tail():
    big = "A" * 100 + "M" * (MAX_TOOL_RESULT_CHARS * 2) + "C" * 100
    out = bound_tool_result(big)
    assert len(out) < len(big)
    assert out.startswith("A")
    assert out.rstrip().endswith("C")
    assert "elided" in out


def test_bound_tool_result_idempotent():
    big = "X" * (MAX_TOOL_RESULT_CHARS * 2)
    once = bound_tool_result(big)
    assert bound_tool_result(once) == once


def test_anthropic_tool_result_carries_is_error():
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "name": "X", "input": {}}]},
        {"role": "tool", "tool_call_id": "t1", "name": "X", "content": "Error: boom", "is_error": True},
    ]
    tr = messages_to_anthropic(msgs)[-1]["content"][0]
    assert tr["type"] == "tool_result"
    assert tr["is_error"] is True


def test_anthropic_tool_result_no_flag_on_success():
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "name": "X", "input": {}}]},
        {"role": "tool", "tool_call_id": "t1", "name": "X", "content": "ok"},
    ]
    tr = messages_to_anthropic(msgs)[-1]["content"][0]
    assert "is_error" not in tr


def test_anthropic_thinking_block_only_with_signature():
    # No signature -> the thinking block must be omitted (replay would 400).
    no_sig = [{"role": "assistant", "content": "hi", "thinking": "reasoning", "tool_calls": []}]
    blocks = messages_to_anthropic(no_sig)[0]["content"]
    assert all(b["type"] != "thinking" for b in blocks)

    # With signature -> thinking block present and FIRST.
    with_sig = [
        {"role": "assistant", "content": "hi", "thinking": "reasoning", "thinking_signature": "sig", "tool_calls": []}
    ]
    blocks2 = messages_to_anthropic(with_sig)[0]["content"]
    assert blocks2[0]["type"] == "thinking"
    assert blocks2[0]["signature"] == "sig"
    assert blocks2[0]["thinking"] == "reasoning"


def test_anthropic_roundtrip_preserves_is_error_and_thinking():
    anthropic_msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "r", "signature": "sig"},
                {"type": "text", "text": "hello"},
                {"type": "tool_use", "id": "t1", "name": "X", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Error: x", "is_error": True}],
        },
    ]
    neutral = messages_from_anthropic(anthropic_msgs)
    asst = neutral[0]
    assert asst["thinking"] == "r"
    assert asst["thinking_signature"] == "sig"
    tool = neutral[1]
    assert tool["is_error"] is True
