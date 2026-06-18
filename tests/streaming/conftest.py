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
"""Shared fixtures for streaming-loop integration tests.

Provides :class:`FakeLLMBuilder` — a test helper that configures canned
LLM responses for the streaming loop without requiring real API keys or
network access. The fixture monkeypatches
:func:`xerxes.streaming.loop._stream_llm` so the entire loop machinery
runs against pre-configured responses.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
from xerxes.streaming import loop as loop_module
from xerxes.streaming.events import TextChunk, ThinkingChunk


class FakeLLMBuilder:
    """Configures canned LLM responses for the streaming loop.

    Each call to the patched ``_stream_llm`` pops the next response from
    the queue. Responses are lists of yielded items: :class:`TextChunk`,
    :class:`ThinkingChunk`, or plain dicts (``{"tool_calls": [...],
    "in_tokens": N, "out_tokens": N}``).

    Example::

        fake = FakeLLMBuilder()
        fake.add_response([
            TextChunk("Let me check."),
            {"tool_calls": [{"id": "c1", "name": "ReadFile",
                             "input": {"file_path": "x.py"}}],
             "in_tokens": 50, "out_tokens": 10},
        ])
        fake.add_response([TextChunk("Done. The file exists.")])
    """

    def __init__(self) -> None:
        self._responses: list[list[Any]] = []
        self._call_count = 0

    def add_response(self, items: list[Any]) -> FakeLLMBuilder:
        """Queue the next LLM response as a list of yielded items."""
        self._responses.append(items)
        return self

    def add_text(self, text: str, in_tokens: int = 10, out_tokens: int = 5) -> FakeLLMBuilder:
        """Queue a simple text-only response with no tool calls."""
        return self.add_response([TextChunk(text), {"tool_calls": [], "in_tokens": in_tokens, "out_tokens": out_tokens}])

    def add_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any] | None = None,
        call_id: str = "call_test",
        text_before: str = "",
        in_tokens: int = 50,
        out_tokens: int = 20,
    ) -> FakeLLMBuilder:
        """Queue a response that requests a single tool call."""
        items: list[Any] = []
        if text_before:
            items.append(TextChunk(text_before))
        items.append(
            {
                "tool_calls": [{"id": call_id, "name": tool_name, "input": tool_input or {}}],
                "in_tokens": in_tokens,
                "out_tokens": out_tokens,
            }
        )
        return self.add_response(items)

    def add_thinking(self, thinking: str, text: str, in_tokens: int = 10, out_tokens: int = 15) -> FakeLLMBuilder:
        """Queue a response with reasoning content followed by visible text."""
        return self.add_response(
            [
                ThinkingChunk(thinking),
                TextChunk(text),
                {"tool_calls": [], "in_tokens": in_tokens, "out_tokens": out_tokens},
            ]
        )

    @property
    def call_count(self) -> int:
        return self._call_count

    def _fake_stream(
        self,
        model: str,
        provider_type: str,
        system: str,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> Generator[Any, None, None]:
        self._call_count += 1
        if not self._responses:
            yield TextChunk("[Fake LLM exhausted]")
            yield {"tool_calls": [], "in_tokens": 0, "out_tokens": 0}
            return
        response = self._responses.pop(0)
        yield from response


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLMBuilder:
    """Patch ``_stream_llm`` to return canned responses.

    Returns a :class:`FakeLLMBuilder`; configure responses with
    ``fake_llm.add_text(...)`` / ``fake_llm.add_tool_call(...)`` before
    running the loop.
    """
    builder = FakeLLMBuilder()
    monkeypatch.setattr(loop_module, "_stream_llm", builder._fake_stream)
    return builder
