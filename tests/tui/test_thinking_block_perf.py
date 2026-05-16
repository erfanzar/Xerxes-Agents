# Copyright 2026 Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
"""Cover the ThinkingBlock perf-hardening: O(1) appends + bounded raw buffer.

Reasoning models (Kimi-for-coding, R1-style chains) routinely stream
hundreds of KBs per turn. The pre-fix block stored everything in a single
``+=``-built string and rebuilt token-count on every render. These tests
pin: appends are O(1) per chunk, the raw buffer compacts to its char
limit, ``raw_text`` materialises lazily, and the cached ``_token_count``
counter matches what the old ``.split()``-per-render approach produced.
"""

from __future__ import annotations

import time

from xerxes.tui.blocks import _ThinkingBlock


def test_append_is_o1_per_chunk():
    block = _ThinkingBlock(block_id="t")
    n = 20_000
    start = time.monotonic()
    for _ in range(n):
        block.append("token ")
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"thinking append regressed: {elapsed:.3f}s for {n} chunks"


def test_raw_buffer_capped_after_long_run():
    block = _ThinkingBlock(block_id="t")
    cap = _ThinkingBlock.RAW_BUFFER_CHAR_LIMIT
    # 4× the cap; the compaction inside ``append`` must keep it bounded.
    for _ in range(cap // 10 * 4):
        block.append("0123456789")
    materialised = block.raw_text
    assert len(materialised) <= cap


def test_token_count_increments_incrementally():
    block = _ThinkingBlock(block_id="t")
    block.append("hello world ")
    block.append("foo bar baz")
    assert block._token_count == 5


def test_append_returns_committed_paragraphs():
    block = _ThinkingBlock(block_id="t")
    assert block.append("first") == []
    committed = block.append("\n\nsecond paragraph")
    assert committed == ["first"]
    assert "second paragraph" in block.raw_text


def test_compose_renders_token_count_without_rescanning():
    block = _ThinkingBlock(block_id="t")
    for _ in range(100):
        block.append("word ")
    rendered = block.compose()
    text = str(rendered.value if hasattr(rendered, "value") else rendered)
    assert "100 tokens" in text
