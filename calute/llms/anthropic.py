
# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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

"""Anthropic Claude LLM implementation."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from .base import BaseLLM, LLMConfig

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    httpx = None


class AnthropicLLM(BaseLLM):
    """Anthropic Claude LLM implementation."""

    def __init__(self, config: LLMConfig | None = None, version: str = "2023-06-01", **kwargs):
        """Initialize Anthropic LLM.

        Args:
            config: LLM configuration
            version: Anthropic API version
            **kwargs: Additional configuration parameters
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
        self.client = None
        super().__init__(config)

    def _initialize_client(self) -> None:
        """Initialize the Anthropic HTTP client."""
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided")

        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={
                "anthropic-version": self.version,
                "x-api-key": api_key,
                "content-type": "application/json",
            },
            timeout=self.config.timeout,
        )

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
        """Generate completion using Anthropic Claude API.

        Args:
            prompt: Text prompt or list of messages
            model: Model override
            temperature: Temperature override
            max_tokens: Max tokens override
            top_p: Top-p override
            stop: Stop sequences override
            stream: Stream override
            **kwargs: Additional Anthropic-specific parameters

        Returns:
            Anthropic completion response
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
                return await self._stream_completion(payload)
            else:
                response = await self.client.post("/v1/messages", json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Anthropic API request failed: {e}") from e

    async def _stream_completion(self, payload: dict) -> AsyncIterator[dict]:
        """Stream completion from Anthropic.

        Args:
            payload: Request payload

        Yields:
            Streaming response chunks
        """
        payload["stream"] = True

        async with self.client.stream("POST", "/v1/messages", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data != "[DONE]":
                        yield json.loads(data)

    def _convert_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Convert OpenAI-style messages to Anthropic format.

        Args:
            messages: OpenAI-style messages

        Returns:
            Anthropic-formatted messages
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
        """Extract content from Anthropic response.

        Args:
            response: Anthropic API response

        Returns:
            The text content from the response
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
        """Process streaming response from Anthropic.

        Args:
            response: Streaming response iterator
            callback: Callback for each chunk

        Returns:
            Complete accumulated content
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
        """Stream completion chunks from Anthropic.

        Args:
            response: Anthropic streaming response
            agent: Optional agent for function detection

        Yields:
            Dictionary with streaming chunk information
        """
        buffered_content = ""
        function_calls = []

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

    async def astream_completion(
        self,
        response: Any,
        agent: Any | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async stream completion chunks from Anthropic.

        Args:
            response: Anthropic async streaming response
            agent: Optional agent for function detection

        Yields:
            Dictionary with streaming chunk information
        """
        buffered_content = ""
        function_calls = []

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

    def parse_tool_calls(self, raw_data: Any) -> list[dict[str, Any]]:
        """Parse tool/function calls from Anthropic format.

        Args:
            raw_data: Anthropic tool use data

        Returns:
            Standardized list of tool calls
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

    async def close(self) -> None:
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
