
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

"""Ollama and local LLM implementations."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from .base import BaseLLM, LLMConfig

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    httpx = None


class OllamaLLM(BaseLLM):
    """Ollama local LLM implementation."""

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize Ollama LLM.

        Args:
            config: LLM configuration
            **kwargs: Additional configuration parameters
        """
        if not HAS_HTTPX:
            raise ImportError("httpx library required for Ollama. Install with: pip install httpx")

        if config is None:
            config = LLMConfig(
                model=kwargs.pop("model", "llama2"),
                base_url=kwargs.pop("base_url", "http://localhost:11434"),
                timeout=kwargs.pop("timeout", 120.0),
                **kwargs,
            )

        self.client = None
        super().__init__(config)

    def _initialize_client(self) -> None:
        """Initialize the HTTP client for Ollama."""
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
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
        """Generate completion using Ollama API.

        Args:
            prompt: Text prompt or list of messages
            model: Model override
            temperature: Temperature override
            max_tokens: Max tokens override
            top_p: Top-p override
            stop: Stop sequences override
            stream: Stream override
            **kwargs: Additional Ollama-specific parameters

        Returns:
            Ollama completion response
        """
        use_stream = stream if stream is not None else self.config.stream

        if isinstance(prompt, list):
            endpoint = "/api/chat"
            payload = {
                "model": model or self.config.model,
                "messages": prompt,
                "stream": use_stream,
                "options": {
                    "temperature": temperature if temperature is not None else self.config.temperature,
                    "top_p": top_p if top_p is not None else self.config.top_p,
                    "num_predict": max_tokens if max_tokens is not None else self.config.max_tokens,
                },
            }
        else:
            endpoint = "/api/generate"
            payload = {
                "model": model or self.config.model,
                "prompt": prompt,
                "stream": use_stream,
                "options": {
                    "temperature": temperature if temperature is not None else self.config.temperature,
                    "top_p": top_p if top_p is not None else self.config.top_p,
                    "num_predict": max_tokens if max_tokens is not None else self.config.max_tokens,
                },
            }

        if stop or self.config.stop:
            payload["options"]["stop"] = stop or self.config.stop

        if self.config.top_k:
            payload["options"]["top_k"] = self.config.top_k

        if "options" in kwargs:
            payload["options"].update(kwargs.pop("options"))
        payload.update(kwargs)

        try:
            if use_stream:
                return await self._stream_completion(endpoint, payload)
            else:
                response = await self.client.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama API request failed: {e}") from e

    async def _stream_completion(self, endpoint: str, payload: dict) -> AsyncIterator[dict]:
        """Stream completion from Ollama.

        Args:
            endpoint: API endpoint
            payload: Request payload

        Yields:
            Streaming response chunks
        """
        async with self.client.stream("POST", endpoint, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    yield json.loads(line)

    def extract_content(self, response: Any) -> str:
        """Extract content from Ollama response.

        Args:
            response: Ollama API response

        Returns:
            The text content from the response
        """
        if isinstance(response, dict):
            if "message" in response:
                return response["message"].get("content", "")

            return response.get("response", "")
        return ""

    async def process_streaming_response(
        self,
        response: Any,
        callback: Callable[[str, Any], None],
    ) -> str:
        """Process streaming response from Ollama.

        Args:
            response: Streaming response iterator
            callback: Callback for each chunk

        Returns:
            Complete accumulated content
        """
        accumulated_content = ""

        async for chunk in response:
            if isinstance(chunk, dict):
                if "message" in chunk:
                    content = chunk["message"].get("content", "")

                else:
                    content = chunk.get("response", "")

                if content:
                    accumulated_content += content
                    callback(content, chunk)

        return accumulated_content

    def stream_completion(
        self,
        response: Any,
        agent: Any | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream completion chunks from Ollama.

        Args:
            response: Ollama streaming response
            agent: Optional agent for function detection

        Yields:
            Dictionary with streaming chunk information
        """
        buffered_content = ""

        for chunk in response:
            chunk_data = {
                "content": None,
                "buffered_content": buffered_content,
                "function_calls": [],
                "tool_calls": None,
                "raw_chunk": chunk,
                "is_final": False,
            }

            if isinstance(chunk, dict):
                if "message" in chunk:
                    content = chunk["message"].get("content", "")
                else:
                    content = chunk.get("response", "")

                if content:
                    buffered_content += content
                    chunk_data["content"] = content
                    chunk_data["buffered_content"] = buffered_content

                if chunk.get("done", False):
                    chunk_data["is_final"] = True

            yield chunk_data

    async def astream_completion(
        self,
        response: Any,
        agent: Any | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async stream completion chunks from Ollama.

        Args:
            response: Ollama async streaming response
            agent: Optional agent for function detection

        Yields:
            Dictionary with streaming chunk information
        """
        buffered_content = ""

        async for chunk in response:
            chunk_data = {
                "content": None,
                "buffered_content": buffered_content,
                "function_calls": [],
                "tool_calls": None,
                "raw_chunk": chunk,
                "is_final": False,
            }

            if isinstance(chunk, dict):
                if "message" in chunk:
                    content = chunk["message"].get("content", "")
                else:
                    content = chunk.get("response", "")

                if content:
                    buffered_content += content
                    chunk_data["content"] = content
                    chunk_data["buffered_content"] = buffered_content

                if chunk.get("done", False):
                    chunk_data["is_final"] = True

            yield chunk_data

    def parse_tool_calls(self, raw_data: Any) -> list[dict[str, Any]]:
        """Parse tool/function calls from Ollama format.

        Args:
            raw_data: Ollama tool call data (if supported)

        Returns:
            Standardized list of tool calls
        """

        return []

    async def close(self) -> None:
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()


class LocalLLM(OllamaLLM):
    """Alias for OllamaLLM for backward compatibility."""

    pass
