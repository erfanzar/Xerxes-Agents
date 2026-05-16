# Copyright 2026 Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
"""Lock the StatusRenderer perf contract (caching, bounds, O(n) streaming).

These tests do not measure wall-clock — they assert on the structural
invariants that make the renderer cheap:

- ``_content_lines`` is a bounded deque so a long session can't grow it.
- ``append_streaming`` runs in O(1) per chunk via a list buffer
  (regression bar: 50k tiny appends finish well under a second).
- ``_markup`` is memoised on a state-version key, so repeated paints with
  unchanged state re-use the cached string.
- ``_count_lines`` matches the old regex-based behaviour without allocating
  a stripped copy.
"""

from __future__ import annotations

import time

from xerxes.tui.prompt import StatusRenderer


def test_content_history_is_bounded():
    r = StatusRenderer()
    limit = StatusRenderer.CONTENT_HISTORY_LIMIT
    for i in range(limit + 500):
        r.append_line(f"line {i}")
    assert len(r._content_lines) == limit
    # Oldest entries rolled off; newest still present.
    assert r._content_lines[-1] == f"line {limit + 499}"
    assert r._content_lines[0] == f"line {500}"


def test_append_streaming_is_o1_per_chunk():
    r = StatusRenderer()
    n = 50_000
    chunk = "x"  # 1 char per chunk maximises the old O(n²) cost
    start = time.monotonic()
    for _ in range(n):
        r.append_streaming(chunk)
    elapsed = time.monotonic() - start
    # The pre-fix version was ~3-5s for this; we'll fail loudly if it
    # regresses to anywhere near that. 1s is generous.
    assert elapsed < 1.0, f"streaming append regressed to O(n²): {elapsed:.3f}s for {n} chunks"


def test_streaming_buffer_capped_after_threshold():
    r = StatusRenderer()
    cap = StatusRenderer.STREAMING_BUFFER_CHAR_LIMIT
    # Pump 3× the cap; the buffer must compact down to ≤ cap chars.
    r.append_streaming("a" * (cap * 3))
    assert r._streaming_chars <= cap
    # The retained slice is the tail.
    assert r._streaming_text.endswith("a")


def test_markup_is_cached_when_state_unchanged():
    r = StatusRenderer()
    r.append_line("hello")
    first = r._markup()
    # State did not change between calls — cache must return the SAME object.
    second = r._markup()
    assert first is second
    # Mutation invalidates the cache.
    r.append_line("world")
    third = r._markup()
    assert third is not first


def test_markup_cache_keyed_on_running_state():
    r = StatusRenderer()
    r.append_line("hi")
    idle = r._markup()
    r.set_running(True)
    running = r._markup()
    assert idle != running
    assert "esc to interrupt" in running


def test_count_lines_matches_plain_text_width():
    text = "\x1b[31mhello\x1b[0m\n\x1b[1mworld 12\x1b[0m"
    lines, last_width = StatusRenderer._count_lines(text)
    assert lines == 2
    assert last_width == len("world 12")


def test_count_lines_handles_empty():
    assert StatusRenderer._count_lines("") == (1, 0)


def test_mark_dirty_bumps_version():
    r = StatusRenderer()
    before = r._state_version
    r.append_line("x")
    assert r._state_version > before


def test_setters_no_op_when_value_unchanged():
    """Setters that get the same value shouldn't bump the version (avoids stale invalidations)."""
    r = StatusRenderer()
    r.set_running(False)
    r.set_plan_mode(False)
    baseline = r._state_version
    r.set_running(False)
    r.set_plan_mode(False)
    r.set_queue_count(0)
    r.set_activity_mode("code")
    assert r._state_version == baseline
