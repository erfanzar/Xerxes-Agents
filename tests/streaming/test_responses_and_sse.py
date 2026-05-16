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
"""Tests for streaming.responses_api + streaming.sse."""

from __future__ import annotations

from xerxes.streaming.events import TextChunk, ThinkingChunk
from xerxes.streaming.responses_api import ResponsesEventTranslator
from xerxes.streaming.sse import SSEEvent, SSEParser, parse_sse_stream


class TestResponsesTranslator:
    def test_text_deltas_yielded(self):
        t = ResponsesEventTranslator()
        events = [
            {"type": "response.output_text.delta", "delta": "Hello "},
            {"type": "response.output_text.delta", "delta": "world"},
            {"type": "response.completed", "response": {"usage": {"input_tokens": 10, "output_tokens": 2}}},
        ]
        chunks = list(t.translate(events))
        texts = [c.text for c in chunks if isinstance(c, TextChunk)]
        assert texts == ["Hello ", "world"]
        assert t.usage.input_tokens == 10
        assert t.usage.output_tokens == 2

    def test_reasoning_emits_thinking(self):
        t = ResponsesEventTranslator()
        events = [
            {"type": "response.reasoning.delta", "delta": "let me think"},
            {"type": "response.completed", "response": {"usage": {}}},
        ]
        out = list(t.translate(events))
        assert any(isinstance(c, ThinkingChunk) and c.text == "let me think" for c in out)

    def test_tool_call_assembled(self):
        t = ResponsesEventTranslator()
        events = [
            {"type": "response.output_item.added", "item": {"id": "call_1", "type": "tool_call", "name": "read"}},
            {"type": "response.function_call_arguments.delta", "item_id": "call_1", "delta": '{"path":'},
            {"type": "response.function_call_arguments.delta", "item_id": "call_1", "delta": '"a.txt"}'},
            {"type": "response.output_item.done", "item": {"id": "call_1", "type": "tool_call", "name": "read"}},
            {"type": "response.completed", "response": {"usage": {}}},
        ]
        list(t.translate(events))
        assert len(t.usage.tool_calls) == 1
        tc = t.usage.tool_calls[0]
        assert tc["name"] == "read"
        assert tc["arguments_text"] == '{"path":"a.txt"}'

    def test_cache_token_counts(self):
        t = ResponsesEventTranslator()
        events = [
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 100, "output_tokens": 200, "cache_read_tokens": 500}},
            }
        ]
        list(t.translate(events))
        assert t.usage.cache_read_tokens == 500

    def test_failure_sets_finish_reason(self):
        t = ResponsesEventTranslator()
        list(t.translate([{"type": "response.failed"}]))
        assert t.usage.finish_reason == "error"


class TestSSEParser:
    def test_simple_event(self):
        events = list(parse_sse_stream(["data: hello\n\n"]))
        assert events == [SSEEvent(event="message", data="hello")]

    def test_multiline_data(self):
        events = list(parse_sse_stream(["data: line1\n", "data: line2\n\n"]))
        assert events[0].data == "line1\nline2"

    def test_custom_event_type(self):
        events = list(parse_sse_stream(["event: tool_call\n", "data: foo\n\n"]))
        assert events[0].event == "tool_call"

    def test_last_event_id_tracked(self):
        p = SSEParser()
        p.feed("id: e-42\n")
        p.feed("data: x\n\n")
        p.drain()
        assert p.last_event_id == "e-42"

    def test_retry_field(self):
        events = list(parse_sse_stream(["retry: 1500\n", "data: y\n\n"]))
        assert events[0].retry == 1500

    def test_comments_ignored(self):
        events = list(parse_sse_stream([": ping\n", "data: z\n\n"]))
        assert [e.data for e in events] == ["z"]

    def test_split_across_feeds(self):
        p = SSEParser()
        p.feed("data: hel")
        assert p.drain() == []
        p.feed("lo\n\n")
        events = p.drain()
        assert events[0].data == "hello"
