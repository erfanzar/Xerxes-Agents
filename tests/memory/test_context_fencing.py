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
"""Tests for xerxes.memory.context_fencing."""

from xerxes.memory.context_fencing import build_memory_context_block, sanitize_context


class TestSanitizeContext:
    def test_removes_fence_tags(self):
        text = "hello <memory-context> world </memory-context>"
        assert sanitize_context(text) == "hello  world "

    def test_case_insensitive(self):
        text = "<MEMORY-CONTEXT>hi</Memory-Context>"
        assert sanitize_context(text) == "hi"

    def test_no_tags_unchanged(self):
        text = "just normal text"
        assert sanitize_context(text) == "just normal text"


class TestBuildMemoryContextBlock:
    def test_wraps_content(self):
        result = build_memory_context_block("User likes dark mode.")
        assert "<memory-context>" in result
        assert "</memory-context>" in result
        assert "User likes dark mode." in result
        assert "recalled memory context" in result

    def test_empty_returns_empty(self):
        assert build_memory_context_block("") == ""
        assert build_memory_context_block("   ") == ""

    def test_strips_existing_fences(self):
        result = build_memory_context_block("<memory-context>old</memory-context>")
        assert "old" in result
        # The old fence tags should be removed
        assert result.count("<memory-context>") == 1  # Only the new opening tag
        assert result.count("</memory-context>") == 1  # Only the new closing tag
