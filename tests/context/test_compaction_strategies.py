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
"""Tests for xerxes.context.compaction_strategies module."""

from xerxes.context.compaction_strategies import (
    PriorityBasedStrategy,
    SlidingWindowStrategy,
    SmartCompactionStrategy,
    SummarizationStrategy,
    TruncateStrategy,
    get_compaction_strategy,
)
from xerxes.types.function_execution_types import CompactionStrategy


def make_messages(n, content_len=50):
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"Message {i}: " + "x" * content_len})
    return msgs


class TestSlidingWindowStrategy:
    def test_compact_short_conversation(self):
        strategy = SlidingWindowStrategy(target_tokens=10000, model="gpt-4")
        msgs = make_messages(3)
        compacted, stats = strategy.compact(msgs)
        assert stats["strategy"] == "sliding_window"
        assert stats["original_count"] == len(msgs)
        assert len(compacted) <= len(msgs)

    def test_compact_preserves_system(self):
        strategy = SlidingWindowStrategy(target_tokens=10000, model="gpt-4", preserve_system=True)
        msgs = make_messages(5)
        compacted, _stats = strategy.compact(msgs)
        system_msgs = [m for m in compacted if m.get("role") == "system"]
        assert len(system_msgs) >= 1

    def test_compact_preserves_recent(self):
        strategy = SlidingWindowStrategy(target_tokens=10000, model="gpt-4", preserve_recent=2)
        msgs = make_messages(10)
        compacted, _stats = strategy.compact(msgs)
        assert len(compacted) >= 2

    def test_compact_with_tight_budget(self):
        strategy = SlidingWindowStrategy(target_tokens=50, model="gpt-4", preserve_recent=2)
        msgs = make_messages(20, content_len=200)
        _compacted, stats = strategy.compact(msgs)
        assert stats["compacted_count"] <= stats["original_count"]

    def test_compact_no_preserve_recent(self):
        strategy = SlidingWindowStrategy(target_tokens=10000, model="gpt-4", preserve_recent=0)
        msgs = make_messages(5)
        compacted, _stats = strategy.compact(msgs)
        assert len(compacted) > 0


class TestTruncateStrategy:
    def test_compact_within_budget(self):
        strategy = TruncateStrategy(target_tokens=10000, model="gpt-4")
        msgs = make_messages(3)
        compacted, stats = strategy.compact(msgs)
        assert stats["strategy"] == "truncate"
        assert len(compacted) == len(msgs)

    def test_compact_over_budget(self):
        strategy = TruncateStrategy(target_tokens=50, model="gpt-4", preserve_recent=2)
        msgs = make_messages(10, content_len=200)
        _compacted, stats = strategy.compact(msgs)
        assert stats["compacted_count"] <= stats["original_count"]

    def test_compact_long_messages_truncated(self):
        strategy = TruncateStrategy(target_tokens=100, model="gpt-4", preserve_recent=1)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x" * 2000},
        ]
        compacted, _stats = strategy.compact(msgs)
        for m in compacted:
            if m["role"] == "user" and len(m["content"]) > 1100:
                raise AssertionError("Should have been truncated")


class TestPriorityBasedStrategy:
    def test_compact_short(self):
        strategy = PriorityBasedStrategy(target_tokens=10000, model="gpt-4")
        msgs = make_messages(3)
        _compacted, stats = strategy.compact(msgs)
        assert stats["strategy"] == "priority_based"

    def test_compact_returns_messages(self):
        strategy = PriorityBasedStrategy(target_tokens=10000, model="gpt-4", preserve_recent=2)
        msgs = make_messages(10)
        compacted, _stats = strategy.compact(msgs)
        assert len(compacted) > 0

    def test_compact_no_compactable(self):
        strategy = PriorityBasedStrategy(target_tokens=10000, model="gpt-4", preserve_recent=10)
        msgs = make_messages(5)
        _compacted, stats = strategy.compact(msgs)
        assert stats["compacted_count"] == len(msgs)

    def test_default_scorer(self):
        strategy = PriorityBasedStrategy(target_tokens=10000, model="gpt-4")
        score = strategy._default_scorer({"role": "system", "content": "hi"}, 0, None)
        assert 0.0 <= score <= 1.0

    def test_default_scorer_with_tool_calls(self):
        strategy = PriorityBasedStrategy(target_tokens=10000, model="gpt-4")
        msg = {"role": "assistant", "content": "result", "tool_calls": []}
        score = strategy._default_scorer(msg, 5, None)
        assert score > 0.5

    def test_default_scorer_long_content(self):
        strategy = PriorityBasedStrategy(target_tokens=10000, model="gpt-4")
        msg = {"role": "user", "content": "x" * 600}
        score = strategy._default_scorer(msg, 0, None)
        assert score >= 0.6

    def test_custom_scorer(self):
        def custom(msg, idx, meta):
            return 1.0 if msg.get("role") == "user" else 0.0

        strategy = PriorityBasedStrategy(target_tokens=10000, model="gpt-4", priority_scorer=custom)
        msgs = make_messages(5, content_len=20)
        compacted, _stats = strategy.compact(msgs)
        assert len(compacted) > 0


class TestSummarizationStrategy:
    def test_compact_no_llm(self):
        strategy = SummarizationStrategy(llm_client=None, target_tokens=10000, model="gpt-4")
        msgs = make_messages(3)
        _compacted, stats = strategy.compact(msgs)
        assert stats["strategy"] == "summarization"

    def test_compact_no_compactable_msgs(self):
        strategy = SummarizationStrategy(llm_client=None, target_tokens=10000, model="gpt-4", preserve_recent=10)
        msgs = make_messages(5)
        _compacted, stats = strategy.compact(msgs)
        assert stats["summary_created"] is False

    def test_compact_with_compactable(self):
        strategy = SummarizationStrategy(llm_client=None, target_tokens=10000, model="gpt-4", preserve_recent=2)
        msgs = make_messages(10)
        _compacted, stats = strategy.compact(msgs)
        assert stats["summary_created"] is True

    def test_format_conversation(self):
        strategy = SummarizationStrategy(llm_client=None, target_tokens=10000, model="gpt-4")
        msgs = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        result = strategy._format_conversation(msgs)
        assert "User: Hello" in result
        assert "Assistant: Hi" in result

    def test_generate_summary_fallback_short(self):
        strategy = SummarizationStrategy(llm_client=None, target_tokens=10000, model="gpt-4")
        text = "Line 1\nLine 2\nLine 3"
        result = strategy._generate_summary(text)
        assert result == text

    def test_generate_summary_fallback_long(self):
        strategy = SummarizationStrategy(llm_client=None, target_tokens=10000, model="gpt-4")
        lines = [f"Line {i}" for i in range(20)]
        text = "\n".join(lines)
        result = strategy._generate_summary(text)
        assert "Earlier discussion covered:" in result


class TestSmartCompactionStrategy:
    def test_compact_light_compression(self):
        strategy = SmartCompactionStrategy(llm_client=None, target_tokens=10000, model="gpt-4")
        msgs = make_messages(3)
        compacted, stats = strategy.compact(msgs)
        assert "substrategy" in stats
        assert len(compacted) > 0

    def test_compact_heavy_compression(self):
        strategy = SmartCompactionStrategy(llm_client=None, target_tokens=10, model="gpt-4")
        msgs = make_messages(20, content_len=200)
        _compacted, stats = strategy.compact(msgs)
        assert stats["substrategy"] in ("truncate", "summarization", "truncate_light")

    def test_compact_medium_compression(self):
        strategy = SmartCompactionStrategy(llm_client=None, target_tokens=500, model="gpt-4")
        msgs = make_messages(10, content_len=100)
        _compacted, stats = strategy.compact(msgs)
        assert "substrategy" in stats


class TestGetCompactionStrategy:
    def test_get_sliding_window(self):
        strategy = get_compaction_strategy(CompactionStrategy.SLIDING_WINDOW, target_tokens=1000)
        assert isinstance(strategy, SlidingWindowStrategy)

    def test_get_truncate(self):
        strategy = get_compaction_strategy(CompactionStrategy.TRUNCATE, target_tokens=1000)
        assert isinstance(strategy, TruncateStrategy)

    def test_get_priority(self):
        strategy = get_compaction_strategy(CompactionStrategy.PRIORITY_BASED, target_tokens=1000)
        assert isinstance(strategy, PriorityBasedStrategy)

    def test_get_summarize(self):
        strategy = get_compaction_strategy(CompactionStrategy.SUMMARIZE, target_tokens=1000)
        assert isinstance(strategy, SummarizationStrategy)

    def test_get_smart(self):
        strategy = get_compaction_strategy(CompactionStrategy.SMART, target_tokens=1000)
        assert isinstance(strategy, SmartCompactionStrategy)


class TestSeparateMessages:
    def test_basic_separation(self):
        strategy = SlidingWindowStrategy(target_tokens=10000, model="gpt-4", preserve_recent=2)
        msgs = make_messages(5)
        sys_msgs, preserved, compactable = strategy._separate_messages(msgs)
        assert len(sys_msgs) == 1
        assert len(preserved) == 2
        assert len(compactable) == 3

    def test_no_system_message(self):
        strategy = SlidingWindowStrategy(target_tokens=10000, model="gpt-4", preserve_system=False, preserve_recent=2)
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        sys_msgs, _preserved, _compactable = strategy._separate_messages(msgs)
        assert len(sys_msgs) == 0

    def test_few_messages(self):
        strategy = SlidingWindowStrategy(target_tokens=10000, model="gpt-4", preserve_recent=5)
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        _sys_msgs, preserved, compactable = strategy._separate_messages(msgs)
        assert len(compactable) == 0
        assert len(preserved) == 2
