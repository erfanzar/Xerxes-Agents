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
"""Tests for thinking-tag parsing in the streaming loop."""

from xerxes.streaming.events import TextChunk, ThinkingChunk
from xerxes.streaming.loop import _parse_thinking_tags, _ThinkingParser


class TestThinkingParser:
    """Unit tests for _ThinkingParser — stateful streaming parser."""

    def test_plain_text_untouched(self):
        parser = _ThinkingParser()
        out = parser.process("Hello world")
        assert len(out) == 1
        assert isinstance(out[0], TextChunk)
        assert out[0].text == "Hello world"

    def test_thinking_tag_stripped_from_visible(self):
        parser = _ThinkingParser()
        out = parser.process("<think> Let me think about this. </think>Done!")
        assert len(out) == 2
        assert isinstance(out[0], ThinkingChunk)
        assert out[0].text == " Let me think about this. "
        assert isinstance(out[1], TextChunk)
        assert out[1].text == "Done!"

    def test_multiple_thinking_blocks(self):
        parser = _ThinkingParser()
        out = parser.process("A <think> First </think> B <think> Second </think> C")
        assert len(out) == 5
        assert isinstance(out[0], TextChunk)
        assert out[0].text == "A "
        assert isinstance(out[1], ThinkingChunk)
        assert out[1].text == " First "
        assert isinstance(out[2], TextChunk)
        assert out[2].text == " B "
        assert isinstance(out[3], ThinkingChunk)
        assert out[3].text == " Second "
        assert isinstance(out[4], TextChunk)
        assert out[4].text == " C"

    def test_thinking_at_start(self):
        parser = _ThinkingParser()
        out = parser.process("<think>Thinking</think>Visible")
        assert len(out) == 2
        assert isinstance(out[0], ThinkingChunk)
        assert out[0].text == "Thinking"
        assert isinstance(out[1], TextChunk)
        assert out[1].text == "Visible"

    def test_thinking_at_end(self):
        parser = _ThinkingParser()
        out = parser.process("Visible<think>Thinking</think>")
        assert len(out) == 2
        assert isinstance(out[0], TextChunk)
        assert out[0].text == "Visible"
        assert isinstance(out[1], ThinkingChunk)
        assert out[1].text == "Thinking"

    def test_unclosed_thinking_buffered(self):
        parser = _ThinkingParser()
        # First chunk: unclosed tag
        out1 = parser.process("<think> Unclosed thinking")
        assert len(out1) == 0
        # Second chunk: closes it
        out2 = parser.process(" still open </think>Done")
        assert len(out2) == 2
        assert isinstance(out2[0], ThinkingChunk)
        assert "still open " in out2[0].text
        assert isinstance(out2[1], TextChunk)
        assert out2[1].text == "Done"

    def test_empty_thinking_block(self):
        parser = _ThinkingParser()
        out = parser.process("<think></think>Done")
        # Empty thinking blocks are skipped (no content to emit)
        assert len(out) == 1
        assert isinstance(out[0], TextChunk)
        assert out[0].text == "Done"

    def test_text_before_and_after_thinking(self):
        parser = _ThinkingParser()
        out = parser.process("Before<think>Inside</think>After")
        assert len(out) == 3
        assert out[0].text == "Before"
        assert out[1].text == "Inside"
        assert out[2].text == "After"

    def test_sequential_chunks(self):
        parser = _ThinkingParser()
        out1 = parser.process("<think>Think")
        assert len(out1) == 0
        out2 = parser.process("ing a lot</think>Answer")
        assert len(out2) == 2
        assert isinstance(out2[0], ThinkingChunk)
        assert out2[0].text == "Thinking a lot"
        assert isinstance(out2[1], TextChunk)
        assert out2[1].text == "Answer"

    def test_nested_thoughts_not_escaped(self):
        parser = _ThinkingParser()
        # The parser is greedy: finds first </think>, no escape for inner tags.
        out = parser.process("<think> Outer <think> Inner </think> Outer cont </think> End")
        assert len(out) == 2
        assert isinstance(out[0], ThinkingChunk)
        assert out[0].text == " Outer <think> Inner "  # inner tags become content
        assert isinstance(out[1], TextChunk)
        assert out[1].text == " Outer cont </think> End"

    def test_mismatched_think_open_thinking_close(self):
        parser = _ThinkingParser()
        out = parser.process("<think> Mismatched </thinking>Done")
        assert len(out) == 2
        assert isinstance(out[0], ThinkingChunk)
        assert out[0].text == " Mismatched "
        assert isinstance(out[1], TextChunk)
        assert out[1].text == "Done"

    def test_thinking_tag_variant(self):
        parser = _ThinkingParser()
        out = parser.process("<thinking> Using thinking variant </thinking>Done")
        assert len(out) == 2
        assert isinstance(out[0], ThinkingChunk)
        assert out[0].text == " Using thinking variant "
        assert isinstance(out[1], TextChunk)
        assert out[1].text == "Done"

    def test_mixed_tag_variants(self):
        parser = _ThinkingParser()
        out = parser.process("A <think> First </think> B <thinking> Second </thinking> C")
        assert len(out) == 5
        assert isinstance(out[0], TextChunk)
        assert out[0].text == "A "
        assert isinstance(out[1], ThinkingChunk)
        assert out[1].text == " First "
        assert isinstance(out[2], TextChunk)
        assert out[2].text == " B "
        assert isinstance(out[3], ThinkingChunk)
        assert out[3].text == " Second "
        assert isinstance(out[4], TextChunk)
        assert out[4].text == " C"


class TestParseThinkingTags:
    """Tests for the _parse_thinking_tags helper (batch mode)."""

    def test_no_thinking(self):
        text = "Just a normal response."
        visible, thinking = _parse_thinking_tags(text)
        assert visible == "Just a normal response."
        assert thinking == ""

    def test_single_thinking_block(self):
        text = "<think> Let me work this out. </think> The answer is 42."
        visible, thinking = _parse_thinking_tags(text)
        assert visible == "The answer is 42."
        assert thinking == "Let me work this out."

    def test_multiple_thinking_blocks(self):
        text = "<think> First </think>Middle<think> Second </think>End"
        visible, thinking = _parse_thinking_tags(text)
        assert visible == "MiddleEnd"
        assert thinking == "First  Second"  # space from trailing content inside each block

    def test_thinking_with_whitespace(self):
        text = "<think>  Spaced  </think>Result"
        visible, thinking = _parse_thinking_tags(text)
        assert visible == "Result"
        assert thinking == "Spaced"

    def test_multiline_thinking(self):
        text = "<think>\nLine1\nLine2\n</think>Result"
        visible, thinking = _parse_thinking_tags(text)
        assert visible == "Result"
        assert "Line1" in thinking
        assert "Line2" in thinking

    def test_mismatched_tags_batch(self):
        text = "<think> Mismatched </thinking> The answer is 42."
        visible, thinking = _parse_thinking_tags(text)
        assert visible == "The answer is 42."
        assert thinking == "Mismatched"

    def test_thinking_variant_batch(self):
        text = "<thinking> Variant </thinking> Result"
        visible, thinking = _parse_thinking_tags(text)
        assert visible == "Result"
        assert thinking == "Variant"
