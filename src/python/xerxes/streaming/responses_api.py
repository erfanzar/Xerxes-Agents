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
"""Translate OpenAI / OpenRouter Responses-API events into neutral chunks.

The Responses API emits a stream of typed events such as
``response.output_text.delta``, ``response.reasoning.delta``,
``response.output_item.added``, ``response.function_call_arguments.delta``,
``response.output_item.done``, and finally ``response.completed``.

:class:`ResponsesEventTranslator` consumes those raw dicts and yields
:class:`TextChunk` / :class:`ThinkingChunk` chunks while accumulating
tool-call records and usage counters on :attr:`ResponsesEventTranslator.usage`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .events import TextChunk, ThinkingChunk


@dataclass
class ResponsesUsage:
    """Accumulated usage and tool-call summary for a Responses-API stream.

    Attributes:
        input_tokens: Prompt tokens reported by ``response.completed``.
        output_tokens: Completion tokens reported by ``response.completed``.
        cache_read_tokens: Tokens served from cache when reported.
        cache_creation_tokens: Tokens written to a new cache entry.
        tool_calls: Tool-call records assembled from ``output_item`` events.
        finish_reason: Final response status (``"stop"``, ``"error"``, …).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"


class ResponsesEventTranslator:
    """Stateful translator from Responses-API events to neutral stream chunks.

    Holds an in-progress buffer of streaming tool-call argument deltas keyed
    by item id, and a :class:`ResponsesUsage` snapshot updated on completion.
    """

    def __init__(self) -> None:
        """Initialise an empty usage snapshot and tool-call argument buffer."""
        self.usage = ResponsesUsage()
        self._buffer: dict[str, dict[str, Any]] = {}

    def translate(self, events: Iterable[dict[str, Any]]) -> Iterator[TextChunk | ThinkingChunk]:
        """Walk Responses-API events and yield neutral chunks.

        Recognises ``response.output_text.delta`` and ``response.reasoning.delta``
        (yielded immediately as :class:`TextChunk` / :class:`ThinkingChunk`),
        assembles tool calls across ``output_item.added`` /
        ``function_call_arguments.delta`` / ``output_item.done`` events into
        :attr:`usage.tool_calls`, and finalises counters on
        ``response.completed`` (or marks ``finish_reason`` on errors).
        """
        for event in events:
            etype = event.get("type", "")
            if etype == "response.output_text.delta":
                delta = event.get("delta") or event.get("text", "")
                if delta:
                    yield TextChunk(delta)
            elif etype == "response.reasoning.delta":
                delta = event.get("delta") or event.get("text", "")
                if delta:
                    yield ThinkingChunk(delta)
            elif etype == "response.output_item.added":
                item = event.get("item", {})
                if item.get("type") == "tool_call":
                    self._buffer[item.get("id", "")] = {"name": item.get("name", ""), "arguments_text": ""}
            elif etype == "response.function_call_arguments.delta":
                call_id = event.get("item_id", "")
                if call_id in self._buffer:
                    self._buffer[call_id]["arguments_text"] += event.get("delta", "")
            elif etype == "response.output_item.done":
                item = event.get("item", {})
                if item.get("type") == "tool_call":
                    call_id = item.get("id", "")
                    record = self._buffer.pop(call_id, {"name": item.get("name", ""), "arguments_text": ""})
                    args_text = item.get("arguments") or record["arguments_text"]
                    self.usage.tool_calls.append(
                        {"id": call_id, "name": record["name"] or item.get("name", ""), "arguments_text": args_text}
                    )
            elif etype == "response.completed":
                resp = event.get("response", {})
                usage = resp.get("usage", {})
                self.usage.input_tokens = int(usage.get("input_tokens", 0))
                self.usage.output_tokens = int(usage.get("output_tokens", 0))
                self.usage.cache_read_tokens = int(usage.get("cache_read_tokens", 0))
                self.usage.cache_creation_tokens = int(usage.get("cache_creation_tokens", 0))
                self.usage.finish_reason = resp.get("status", "stop")
            elif etype in ("response.failed", "error"):
                self.usage.finish_reason = "error"


__all__ = ["ResponsesEventTranslator", "ResponsesUsage"]
