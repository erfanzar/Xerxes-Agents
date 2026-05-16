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
"""Tests for xerxes.context.advanced_compressor."""

from xerxes.context.advanced_compressor import (
    SUMMARY_PREFIX,
    AdvancedCompressionStrategy,
    _prune_tool_results,
    _summarize_tool_result,
)


class TestSummarizeToolResult:
    """Tests for _summarize_tool_result."""

    def test_terminal_tool(self):
        result = _summarize_tool_result("terminal", '{"command": "npm test"}', '{"exit_code": 0, "output": "..."}')
        assert "terminal" in result
        assert "npm test" in result
        assert "exit 0" in result

    def test_read_file_tool(self):
        result = _summarize_tool_result("read_file", '{"path": "config.py", "offset": 1}', "content here")
        assert "read_file" in result
        assert "config.py" in result

    def test_write_file_tool(self):
        result = _summarize_tool_result("write_file", '{"path": "out.txt", "content": "hello\\nworld"}', "done")
        assert "write_file" in result
        assert "out.txt" in result

    def test_generic_tool(self):
        result = _summarize_tool_result("custom_tool", '{"foo": "bar"}', "some output")
        assert "custom_tool" in result
        assert "foo=bar" in result


class TestPruneToolResults:
    """Tests for _prune_tool_results."""

    def test_short_tool_untouched(self):
        messages = [
            {"role": "tool", "name": "test", "content": "short", "args": {}},
        ]
        pruned = _prune_tool_results(messages)
        assert pruned[0]["content"] == "short"

    def test_long_tool_pruned(self):
        messages = [
            {"role": "tool", "name": "test", "content": "x" * 1000, "args": {}},
        ]
        pruned = _prune_tool_results(messages)
        assert len(pruned[0]["content"]) < 500
        assert "test" in pruned[0]["content"]
        assert pruned[0]["_original_content"] == "x" * 1000

    def test_non_tool_untouched(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        pruned = _prune_tool_results(messages)
        assert pruned == messages


class TestAdvancedCompressionStrategy:
    """Tests for :class:`AdvancedCompressionStrategy`."""

    def test_no_compaction_needed(self):
        strategy = AdvancedCompressionStrategy(target_tokens=100_000)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        _compacted, stats = strategy.compact(messages)
        assert stats["summary_created"] is False
        assert stats["original_count"] == 3

    def test_tool_pruning_happens(self):
        strategy = AdvancedCompressionStrategy(target_tokens=1000, preserve_recent=0, tail_token_budget=100)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Run tests"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "terminal", "arguments": '{"command": "npm test"}'}}],
            },
            {"role": "tool", "name": "terminal", "content": "x" * 2000, "args": {"command": "npm test"}},
            {"role": "assistant", "content": "Tests passed!"},
            {"role": "user", "content": "Great"},
        ]
        compacted, stats = strategy.compact(messages)
        assert stats["tools_pruned"] >= 1
        # Verify the pruned tool appears in output with a summary line
        tool_msg = next(m for m in compacted if m.get("role") == "tool")
        assert "terminal" in tool_msg["content"]
        assert "x" * 10 not in tool_msg["content"]  # Original content was pruned

    def test_summary_has_handoff_framing(self):
        strategy = AdvancedCompressionStrategy(target_tokens=1000, preserve_recent=0, tail_token_budget=10)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Answer 2"},
            {"role": "user", "content": "Question 3"},
            {"role": "assistant", "content": "Answer 3"},
            {"role": "user", "content": "Question 4"},
            {"role": "assistant", "content": "Answer 4"},
        ]
        compacted, _stats = strategy.compact(messages)
        summary_msg = compacted[1]  # After system msg
        assert "handoff" in summary_msg["content"].lower()
        assert SUMMARY_PREFIX in summary_msg["content"]

    def test_iterative_compaction(self):
        strategy = AdvancedCompressionStrategy(target_tokens=1000, preserve_recent=0, tail_token_budget=10)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "A3"},
            {"role": "user", "content": "Q4"},
            {"role": "assistant", "content": "A4"},
        ]
        compacted1, stats1 = strategy.compact(messages)
        assert stats1["compaction_count"] == 1

        # Add more messages and compact again
        messages2 = list(compacted1)
        messages2.extend(
            [
                {"role": "user", "content": "Q5"},
                {"role": "assistant", "content": "A5"},
                {"role": "user", "content": "Q6"},
                {"role": "assistant", "content": "A6"},
            ]
        )
        _compacted2, stats2 = strategy.compact(messages2)
        assert stats2["compaction_count"] == 2

    def test_tail_protection(self):
        strategy = AdvancedCompressionStrategy(target_tokens=1000, preserve_recent=2, tail_token_budget=100)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Old message 1"},
            {"role": "assistant", "content": "Old response 1"},
            {"role": "user", "content": "Old message 2"},
            {"role": "assistant", "content": "Old response 2"},
            {"role": "user", "content": "Recent message 1"},
            {"role": "assistant", "content": "Recent response 1"},
        ]
        _compacted, stats = strategy.compact(messages)
        assert stats["tail_messages"] >= 1

    def test_from_enum(self):
        from xerxes.context import get_compaction_strategy
        from xerxes.types.function_execution_types import CompactionStrategy

        strategy = get_compaction_strategy(CompactionStrategy.ADVANCED, target_tokens=4000)
        assert isinstance(strategy, AdvancedCompressionStrategy)
