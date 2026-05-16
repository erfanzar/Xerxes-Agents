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
"""Anthropic Claude provider adapter built on ``httpx``.

Talks directly to the Messages API rather than the official SDK.
``stream_completion`` / ``astream_completion`` emit normalised chunk
dicts (see :class:`BaseLLM`); Anthropic-specific extras include
thinking-content deltas for extended-thinking models. :data:`ANTHROPIC_CONTEXT_LENGTHS`
is a static fallback for context-window lookup.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from .base import BaseLLM, LLMConfig

ANTHROPIC_CONTEXT_LENGTHS = {
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-3-5-sonnet": 200000,
    "claude-3-5-haiku": 200000,
    "claude-opus-4": 200000,
    "claude-sonnet-4": 200000,
}

httpx: Any
try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    httpx = None


class AnthropicLLM(BaseLLM):
    """Anthropic LLM client using the Messages API.

    Wraps the Anthropic HTTP API to provide completion, streaming,
    and tool call support for Claude models.

    Attributes:
        version: The Anthropic API version string.
        client: The underlying httpx AsyncClient for API requests.
    """

    def __init__(self, config: LLMConfig | None = None, version: str = "2023-06-01", **kwargs):
        """Initialize the Anthropic LLM client.

        Args:
            config: Optional LLM configuration. If None, defaults to
                "claude-3-opus-20240229" with the Anthropic API endpoint.
            version: The Anthropic API version header value.
            **kwargs: Additional configuration fields forwarded to LLMConfig.
        """

        if not HAS_HTTPX:
            raise ImportError("httpx library required for Anthropic. Install with: pip install httpx")

        if config is None:
            config = LLMConfig(
                model=kwargs.pop("model", "claude-3-opus-20240229"),
                api_key=kwargs.pop("api_key", None),
                base_url=kwargs.pop("base_url", "https://api.anthropic.com"),
                **kwargs,
            )

        self.version = version
        self.client: httpx.AsyncClient | None = None
        super().__init__(config)

    def _initialize_client(self) -> None:
        """Initialize the httpx AsyncClient with Anthropic authentication headers.

        Reads the API key from the ANTHROPIC_API_KEY environment variable and
        configures the base URL and headers for all requests.
        """

        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided")

        self.client = httpx.AsyncClient(
            base_url=self.config.base_url or "",
            headers={
                "anthropic-version": self.version,
                "x-api-key": api_key,
                "content-type": "application/json",
            },
            timeout=self.config.timeout,
        )
        self._auto_fetch_model_info()

    async def generate_completion(
        self,
        prompt: str | list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        stream: bool | None = None,
        **kwargs,
    ) -> Any:
        """Generate a completion response from the Anthropic API.

        Args:
            prompt: A string prompt or a list of message dicts to send.
            model: Override the default model for this request.
            temperature: Override sampling temperature.
            max_tokens: Override maximum tokens to generate.
            top_p: Override nucleus sampling threshold.
            stop: Override stop sequences.
            stream: Whether to return a streaming response.
            **kwargs: Additional provider-specific parameters.

        Returns:
            A dict response from the API, or an async iterator of streaming events
            if stream=True.
        """

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = self._convert_messages(prompt)

        payload = {
            "model": model or self.config.model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }

        if top_p is not None or self.config.top_p != 0.95:
            payload["top_p"] = top_p if top_p is not None else self.config.top_p

        if stop or self.config.stop:
            payload["stop_sequences"] = stop or self.config.stop

        payload.update(kwargs)
        payload.update(self.config.extra_params)

        use_stream = stream if stream is not None else self.config.stream

        try:
            if use_stream:
                return self._stream_completion(payload)
            else:
                assert self.client is not None
                response = await self.client.post("/v1/messages", json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Anthropic API request failed: {e}") from e

    async def _stream_completion(self, payload: dict) -> AsyncIterator[dict]:
        """Yield streaming events from the Anthropic API.

        Args:
            payload: The request payload with streaming enabled.

        Yields:
            Parsed JSON dictionaries from each SSE data line.
        """

        payload["stream"] = True
        assert self.client is not None

        async with self.client.stream("POST", "/v1/messages", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data != "[DONE]":
                        yield json.loads(data)

    def _convert_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Convert OpenAI-style messages to Anthropic message format.

        Moves system messages to the first user message or prepends a user
        message containing the system content.

        Args:
            messages: List of OpenAI-style message dicts.

        Returns:
            A list of messages formatted for the Anthropic API.
        """

        converted = []
        system_content = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_content = content
            else:
                if role in ["user", "assistant"]:
                    converted.append({"role": role, "content": content})

        if system_content and converted:
            if converted[0]["role"] == "user":
                converted[0]["content"] = f"{system_content}\n\n{converted[0]['content']}"
            else:
                converted.insert(0, {"role": "user", "content": system_content})

        return converted

    def extract_content(self, response: Any) -> str:
        """Extract text content from an Anthropic API response.

        Args:
            response: The raw API response dict.

        Returns:
            The concatenated text from all "text" type content blocks.
        """

        if isinstance(response, dict):
            content = response.get("content", [])
            if content and isinstance(content, list):
                text_parts = []
                for block in content:
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                return "".join(text_parts)
        return ""

    async def process_streaming_response(
        self,
        response: Any,
        callback: Callable[[str, Any], None],
    ) -> str:
        """Process a streaming response and invoke a callback for each text chunk.

        Args:
            response: The streaming response iterator from generate_completion.
            callback: Callable invoked with (text_chunk, raw_chunk) for each
                "content_block_delta" event.

        Returns:
            The concatenated text of all streamed chunks.
        """

        accumulated_content = ""

        async for chunk in response:
            if chunk.get("type") == "content_block_delta":
                delta = chunk.get("delta", {})
                if text := delta.get("text"):
                    accumulated_content += text
                    callback(text, chunk)

        return accumulated_content

    def stream_completion(
        self,
        response: Any,
        agent: Any | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield structured chunks from a streaming Anthropic response.

        Args:
            response: The raw streaming response from the API.
            agent: Optional agent context (unused).

        Yields:
            Dictionaries containing "content", "buffered_content", "function_calls",
            "is_final", and "raw_chunk".
        """

        buffered_content = ""
        function_calls: list[dict[str, Any]] = []

        for event in response:
            chunk_data = {
                "content": None,
                "buffered_content": buffered_content,
                "function_calls": [],
                "tool_calls": None,
                "raw_chunk": event,
                "is_final": False,
            }

            event_type = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)

            if event_type == "content_block_delta":
                delta = event.get("delta", {}) if isinstance(event, dict) else getattr(event, "delta", {})
                text = delta.get("text", "") if isinstance(delta, dict) else getattr(delta, "text", "")
                if text:
                    buffered_content += text
                    chunk_data["content"] = text
                    chunk_data["buffered_content"] = buffered_content
            elif event_type == "message_stop":
                chunk_data["is_final"] = True
                chunk_data["function_calls"] = function_calls
            elif event_type == "tool_use":
                name = event.get("name") if isinstance(event, dict) else getattr(event, "name", None)
                input_data = event.get("input") if isinstance(event, dict) else getattr(event, "input", None)
                if name:
                    function_calls.append(
                        {
                            "id": event.get("id") if isinstance(event, dict) else getattr(event, "id", None),
                            "name": name,
                            "arguments": json.dumps(input_data) if input_data else "",
                        }
                    )

            yield chunk_data

    def astream_completion(
        self,
        response: Any,
        agent: Any | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield structured chunks from a streaming Anthropic response asynchronously.

        Args:
            response: The raw async streaming response from the API.
            agent: Optional agent context (unused).

        Yields:
            Dictionaries containing "content", "buffered_content", "function_calls",
            "is_final", and "raw_chunk".
        """

        async def _gen() -> AsyncIterator[dict[str, Any]]:
            """Internal async generator that yields structured chunks.

            Yields:
                Dictionaries containing "content", "buffered_content", "function_calls",
                "is_final", and "raw_chunk".
            """
            buffered_content = ""
            function_calls: list[dict[str, Any]] = []

            async for event in response:
                chunk_data = {
                    "content": None,
                    "buffered_content": buffered_content,
                    "function_calls": [],
                    "tool_calls": None,
                    "raw_chunk": event,
                    "is_final": False,
                }

                event_type = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)

                if event_type == "content_block_delta":
                    delta = event.get("delta", {}) if isinstance(event, dict) else getattr(event, "delta", {})
                    text = delta.get("text", "") if isinstance(delta, dict) else getattr(delta, "text", "")
                    if text:
                        buffered_content += text
                        chunk_data["content"] = text
                        chunk_data["buffered_content"] = buffered_content
                elif event_type == "message_stop":
                    chunk_data["is_final"] = True
                    chunk_data["function_calls"] = function_calls
                elif event_type == "tool_use":
                    name = event.get("name") if isinstance(event, dict) else getattr(event, "name", None)
                    input_data = event.get("input") if isinstance(event, dict) else getattr(event, "input", None)
                    if name:
                        function_calls.append(
                            {
                                "id": event.get("id") if isinstance(event, dict) else getattr(event, "id", None),
                                "name": name,
                                "arguments": json.dumps(input_data) if input_data else "",
                            }
                        )

                yield chunk_data

        return _gen()

    def parse_tool_calls(self, raw_data: Any) -> list[dict[str, Any]]:
        """Extract tool calls from an Anthropic API response.

        Args:
            raw_data: The raw response dict containing content blocks.

        Returns:
            A list of tool call dicts with "id", "name", and "arguments" keys.
        """
        tool_calls = []
        if isinstance(raw_data, dict) and "content" in raw_data:
            for block in raw_data["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id"),
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        }
                    )
        return tool_calls

    def fetch_model_info(self) -> dict[str, Any]:
        """Return model metadata based on the configured model name.

        Looks up the model prefix in ANTHROPIC_CONTEXT_LENGTHS to determine
        the maximum context window size.

        Returns:
            A dict containing "max_model_len" if a match is found.
        """
        model = self.config.model
        for prefix, context_len in ANTHROPIC_CONTEXT_LENGTHS.items():
            if model.startswith(prefix):
                return {"max_model_len": context_len}
        return {}

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self.client:
            await self.client.aclose()
