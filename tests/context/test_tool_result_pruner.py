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
"""Tests for xerxes.context.tool_result_pruner."""

from __future__ import annotations

from xerxes.context.tool_result_pruner import prune_messages, prune_tool_result


class TestPruneToolResult:
    def test_short_text_unchanged(self) -> None:
        out, did = prune_tool_result("hello world")
        assert out == "hello world"
        assert did is False

    def test_non_string_unchanged(self) -> None:
        out, did = prune_tool_result({"key": "value"})
        assert out == {"key": "value"}
        assert did is False

    def test_long_text_truncated(self) -> None:
        text = "\n".join(f"line-{i}" for i in range(200))
        out, did = prune_tool_result(text, max_chars=100, head_lines=5, tail_lines=2)
        assert did is True
        assert "line-0" in out
        assert "line-199" in out
        assert "lines omitted" in out
        # Should be smaller than the original.
        assert len(out) < len(text)

    def test_binary_content_replaced(self) -> None:
        # Generate content with >30% non-printable bytes.
        binary = "".join(chr(i % 32) for i in range(8_000))  # mostly control chars
        out, did = prune_tool_result(binary, max_chars=1_000)
        assert did is True
        assert "binary content" in out

    def test_exactly_at_max_chars_unchanged(self) -> None:
        text = "x" * 100
        out, did = prune_tool_result(text, max_chars=100)
        assert did is False
        assert out == text


class TestPruneMessages:
    def test_empty_list(self) -> None:
        out, count = prune_messages([])
        assert out == []
        assert count == 0

    def test_only_tool_messages_pruned(self) -> None:
        long = "y" * 50_000
        msgs = [
            {"role": "user", "content": long},
            {"role": "tool", "content": long, "tool_call_id": "1"},
            {"role": "user", "content": "tail"},
            {"role": "assistant", "content": long},
        ]
        out, count = prune_messages(msgs, protect_last=0)
        # Only the tool message should change.
        assert count == 1
        assert out[0]["content"] == long  # user untouched
        assert "omitted" in out[1]["content"]  # tool pruned
        assert out[3]["content"] == long  # assistant untouched

    def test_last_messages_protected(self) -> None:
        long = "y" * 50_000
        msgs = [
            {"role": "tool", "content": long, "tool_call_id": "1"},
            {"role": "tool", "content": long, "tool_call_id": "2"},
            {"role": "tool", "content": long, "tool_call_id": "3"},
        ]
        out, count = prune_messages(msgs, protect_last=2)
        # Only the first one (outside protected window) is pruned.
        assert count == 1
        assert "omitted" in out[0]["content"]
        assert out[1]["content"] == long
        assert out[2]["content"] == long

    def test_original_messages_not_mutated(self) -> None:
        long = "y" * 50_000
        msg = {"role": "tool", "content": long, "tool_call_id": "1"}
        msgs = [msg]
        prune_messages(msgs, protect_last=0)
        assert msgs[0]["content"] == long  # original list/dict untouched
