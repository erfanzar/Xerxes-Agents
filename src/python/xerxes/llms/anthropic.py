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
"""Anthropic module for Xerxes.

Exports:
    - ANTHROPIC_CONTEXT_LENGTHS
    - AnthropicLLM"""

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
    """Anthropic llm.

    Inherits from: BaseLLM
    """

    def __init__(self, config: LLMConfig | None = None, version: str = "2023-06-01", **kwargs):
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            config (LLMConfig | None, optional): IN: config. Defaults to None. OUT: Consumed during execution.
            version (str, optional): IN: version. Defaults to '2023-06-01'. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            Any: OUT: Result of the operation."""

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
        """Internal helper to initialize client.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

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
        """Asynchronously Generate completion.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            prompt (str | list[dict[str, str]]): IN: prompt. OUT: Consumed during execution.
            model (str | None, optional): IN: model. Defaults to None. OUT: Consumed during execution.
            temperature (float | None, optional): IN: temperature. Defaults to None. OUT: Consumed during execution.
            max_tokens (int | None, optional): IN: max tokens. Defaults to None. OUT: Consumed during execution.
            top_p (float | None, optional): IN: top p. Defaults to None. OUT: Consumed during execution.
            stop (list[str] | None, optional): IN: stop. Defaults to None. OUT: Consumed during execution.
            stream (bool | None, optional): IN: stream. Defaults to None. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            Any: OUT: Result of the operation."""

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
        """Asynchronously Internal helper to stream completion.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            payload (dict): IN: payload. OUT: Consumed during execution.
        Returns:
            AsyncIterator[dict]: OUT: Result of the operation."""

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
        """Internal helper to convert messages.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            messages (list[dict[str, str]]): IN: messages. OUT: Consumed during execution.
        Returns:
            list[dict[str, str]]: OUT: Result of the operation."""

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
        """Extract content.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            response (Any): IN: response. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

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
        """Asynchronously Process streaming response.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            response (Any): IN: response. OUT: Consumed during execution.
            callback (Callable[[str, Any], None]): IN: callback. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

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
        """Stream completion.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            response (Any): IN: response. OUT: Consumed during execution.
            agent (Any | None, optional): IN: agent. Defaults to None. OUT: Consumed during execution.
        Returns:
            Iterator[dict[str, Any]]: OUT: Result of the operation."""

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
        """Astream completion.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            response (Any): IN: response. OUT: Consumed during execution.
            agent (Any | None, optional): IN: agent. Defaults to None. OUT: Consumed during execution.
        Returns:
            AsyncIterator[dict[str, Any]]: OUT: Result of the operation."""

        async def _gen() -> AsyncIterator[dict[str, Any]]:
            """Asynchronously Internal helper to gen.

            Returns:
                AsyncIterator[dict[str, Any]]: OUT: Result of the operation."""
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
        """Parse tool calls.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            raw_data (Any): IN: raw data. OUT: Consumed during execution.
        Returns:
            list[dict[str, Any]]: OUT: Result of the operation."""

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
        """Fetch model info.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

        model = self.config.model
        for prefix, context_len in ANTHROPIC_CONTEXT_LENGTHS.items():
            if model.startswith(prefix):
                return {"max_model_len": context_len}
        return {}

    async def close(self) -> None:
        """Asynchronously Close.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        if self.client:
            await self.client.aclose()
