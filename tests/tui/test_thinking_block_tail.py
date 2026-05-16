# Copyright 2026 Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
"""Cover the thinking-block tail truncation so reasoning traces stay bounded.

Models like Kimi-for-coding can stream tens of KBs of reasoning per turn.
Without truncation the thinking block trampled the rest of the viewport;
this test pins the rule: at most ``THINKING_TAIL_LINES`` lines and
``THINKING_TAIL_CHARS`` chars, prefixed with ``…`` when anything was cut.
"""

from __future__ import annotations

from xerxes.tui.blocks import _ThinkingBlock


def test_tail_no_truncation_when_short():
    out = _ThinkingBlock._tail("hi there", max_lines=3, max_chars=200)
    assert out == "hi there"


def test_tail_truncates_to_last_n_lines():
    text = "line1\nline2\nline3\nline4\nline5"
    out = _ThinkingBlock._tail(text, max_lines=3, max_chars=200)
    assert out == "…line3\nline4\nline5"


def test_tail_truncates_to_last_n_chars():
    text = "abcdefghij" * 30  # 300 chars, single line
    out = _ThinkingBlock._tail(text, max_lines=3, max_chars=200)
    assert out.startswith("…")
    body = out[1:]
    assert len(body) <= 200
    assert text.endswith(body)


def test_tail_combines_char_then_line_truncation():
    text = "x" * 500 + "\n" + "\n".join(f"line{i}" for i in range(10))
    out = _ThinkingBlock._tail(text, max_lines=3, max_chars=200)
    assert out.startswith("…")
    assert out.count("\n") <= 2  # 3 lines = 2 newlines


def test_compose_renders_truncated_tail():
    block = _ThinkingBlock(block_id="b")
    block._raw = "filler " * 200 + "FINAL_TOKEN"
    rendered = block.compose()
    # ANSI wraps the formatted output; the rendered text must still contain
    # the latest content but not the early "filler".
    text = str(rendered.value if hasattr(rendered, "value") else rendered)
    assert "FINAL_TOKEN" in text
    assert "…" in text
