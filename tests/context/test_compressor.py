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
"""Tests for xerxes.context.compressor."""

from __future__ import annotations

import pytest
from xerxes.context.compressor import (
    COMPACTION_REFERENCE_PREFIX,
    ContextCompressor,
    naive_summarizer,
)


def _msgs(n: int, role: str = "user", prefix: str = "msg") -> list[dict]:
    return [{"role": role, "content": f"{prefix}-{i}"} for i in range(n)]


class TestNaiveSummarizer:
    def test_empty_returns_empty(self) -> None:
        assert naive_summarizer([], 1000) == ""

    def test_summarizes_first_line_per_message(self) -> None:
        out = naive_summarizer(
            [
                {"role": "user", "content": "hello\nworld"},
                {"role": "assistant", "content": "hi"},
            ],
            1000,
        )
        assert "user: hello" in out
        assert "assistant: hi" in out

    def test_handles_non_string(self) -> None:
        out = naive_summarizer([{"role": "tool", "content": ["a", "b"]}], 1000)
        assert "tool:" in out

    def test_truncates_long_first_line(self) -> None:
        long = "x" * 500
        out = naive_summarizer([{"role": "user", "content": long}], 1000)
        assert "…" in out


class TestThresholdAndShouldCompact:
    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            ContextCompressor(threshold=0)
        with pytest.raises(ValueError):
            ContextCompressor(threshold=1.5)

    def test_invalid_protected_raises(self) -> None:
        with pytest.raises(ValueError):
            ContextCompressor(protect_first=-1)

    def test_threshold_tokens_math(self) -> None:
        c = ContextCompressor(threshold=0.5, context_window=10_000)
        assert c.threshold_tokens() == 5_000

    def test_should_compact_below_returns_false(self) -> None:
        c = ContextCompressor(threshold=0.9, context_window=1_000_000)
        assert c.should_compact(_msgs(3)) is False


class TestCompressionAlgorithm:
    def _setup(self, **overrides) -> ContextCompressor:
        defaults = dict(
            threshold=0.01,  # force compression for tiny test messages
            context_window=1_000,
            protect_first=1,
            protect_last=2,
            summary_min_tokens=10,
            summary_max_tokens=1_000,
            summarizer=lambda messages, budget: f"SUMMARY of {len(messages)} messages",
        )
        defaults.update(overrides)
        return ContextCompressor(**defaults)

    def test_empty_messages(self) -> None:
        c = self._setup()
        res = c.compress([])
        assert res.compressed is False
        assert res.messages == []

    def test_protects_first_and_last(self) -> None:
        c = self._setup(protect_first=1, protect_last=2)
        ms = _msgs(10)
        res = c.compress(ms)
        # head + summary + tail
        assert res.compressed is True
        # Protected head: first user message preserved verbatim.
        assert res.messages[0]["content"] == "msg-0"
        # Protected tail: last two preserved.
        assert res.messages[-2]["content"] == "msg-8"
        assert res.messages[-1]["content"] == "msg-9"
        # Summary in between with the reference prefix.
        summary = res.messages[1]
        assert summary["role"] == "user"
        assert summary["content"].startswith(COMPACTION_REFERENCE_PREFIX)

    def test_compressed_count_equals_middle_length(self) -> None:
        c = self._setup(protect_first=1, protect_last=1)
        ms = _msgs(10)
        res = c.compress(ms)
        # Middle slice = 8 messages (between first and last).
        assert res.compressed_count == 8

    def test_no_middle_returns_unchanged(self) -> None:
        c = self._setup(protect_first=2, protect_last=2)
        # 4 messages → protected head=2, tail=2, middle=0.
        ms = _msgs(4)
        res = c.compress(ms)
        # Even though threshold says compact, there's no middle to summarize.
        assert res.compressed_count == 0

    def test_iterative_merge(self) -> None:
        """A second compression should merge with the prior summary, not stack a new one."""
        c = self._setup(protect_first=1, protect_last=2)
        first = c.compress(_msgs(8))
        assert first.compressed is True
        # The result list looks like: [user-0, summary, user-6, user-7].
        # Feed it back in as if more turns happened.
        more = first.messages + _msgs(8, prefix="next")
        second = c.compress(more)
        # The new summary should reference the old (iterative).
        assert second.metadata.get("strategy") == "iterative"
        # Only one summary block exists in the output (no duplicate prefix).
        prefix_count = sum(
            1
            for m in second.messages
            if isinstance(m.get("content"), str) and m["content"].startswith(COMPACTION_REFERENCE_PREFIX)
        )
        assert prefix_count == 1

    def test_summary_budget_min_max(self) -> None:
        c = self._setup(summary_min_tokens=100, summary_max_tokens=200)
        assert c._summary_budget(0) == 100
        assert c._summary_budget(500) == 100  # 20% of 500 = 100; respects min
        assert c._summary_budget(5_000) == 200  # 20% would be 1000; respects max

    def test_prune_only_path(self) -> None:
        """When pre-pruning alone gets us below threshold, no summarization needed."""
        c = self._setup(
            threshold=0.5,
            context_window=200_000,
            protect_last=1,
        )
        # Build messages with one huge tool result and tiny tail.
        huge = "x" * 100_000  # well above max_chars
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "calling tool"},
            {"role": "tool", "content": huge, "tool_call_id": "1"},
            {"role": "user", "content": "thanks"},
        ]
        res = c.compress(messages)
        assert res.compressed is True
        # The tool result was pruned, summary not needed.
        assert res.metadata.get("strategy") == "prune-only"
        assert res.pruned_tool_results >= 1


class TestCompressionResultFields:
    def test_token_counts_populated(self) -> None:
        c = ContextCompressor(
            threshold=0.01,
            context_window=100,
            protect_first=1,
            protect_last=1,
            summarizer=lambda m, b: "S",
        )
        res = c.compress(_msgs(5))
        assert res.tokens_before > 0
        assert res.tokens_after > 0

    def test_metadata_strategy_set(self) -> None:
        c = ContextCompressor(
            threshold=0.01,
            context_window=100,
            protect_first=1,
            protect_last=1,
            summarizer=lambda m, b: "S",
        )
        res = c.compress(_msgs(5))
        assert "strategy" in res.metadata
