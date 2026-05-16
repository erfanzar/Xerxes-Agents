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
"""Ollama provider adapter for self-hosted local models.

:class:`OllamaLLM` talks directly to an Ollama HTTP daemon (default
``http://localhost:11434``) over ``httpx``. :class:`LocalLLM` is a
thin alias used by ``create_llm("local")``. Streaming emits the
normalised chunk dicts described in :class:`BaseLLM`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from .base import BaseLLM, LLMConfig

httpx: Any
try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    httpx = None


class OllamaLLM(BaseLLM):
    """Ollama LLM client using the Ollama REST API.

    Connects to a local or remote Ollama server to provide completion
    and streaming support for models served via Ollama.

    Attributes:
        client: The underlying httpx AsyncClient for API requests.
    """

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize the Ollama LLM client.

        Args:
            config: Optional LLM configuration. If None, defaults to "llama2"
                with the local Ollama endpoint (http://localhost:11434).
            **kwargs: Additional configuration fields forwarded to LLMConfig.
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

        self.client: httpx.AsyncClient | None = None
        super().__init__(config)

    def _initialize_client(self) -> None:
        """Initialize the httpx AsyncClient for the Ollama API endpoint."""

        self.client = httpx.AsyncClient(
            base_url=self.config.base_url or "",
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
        """Generate a completion response from the Ollama API.

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
            A dict response from the API, or an async iterator if stream=True.
        """

        use_stream = stream if stream is not None else self.config.stream

        payload: dict[str, Any]
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
                return self._stream_completion(endpoint, payload)
            else:
                assert self.client is not None
                response = await self.client.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama API request failed: {e}") from e

    async def _stream_completion(self, endpoint: str, payload: dict) -> AsyncIterator[dict]:
        """Yield streaming response lines from the Ollama API.

        Args:
            endpoint: The API endpoint to call.
            payload: The request payload with streaming enabled.

        Yields:
            Parsed JSON dictionaries from each line of the response.
        """

        assert self.client is not None
        async with self.client.stream("POST", endpoint, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    yield json.loads(line)

    def extract_content(self, response: Any) -> str:
        """Extract text content from an Ollama API response.

        Args:
            response: The raw API response dict.

        Returns:
            The extracted text from "message.content" or "response" fields.
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
        """Process a streaming response and invoke a callback for each text chunk.

        Args:
            response: The streaming response iterator from generate_completion.
            callback: Callable invoked with (text_chunk, raw_chunk) for each chunk.

        Returns:
            The concatenated text of all streamed chunks.
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
        """Yield structured chunks from a streaming Ollama response.

        Args:
            response: The raw streaming response from the API.
            agent: Optional agent context (unused).

        Yields:
            Dictionaries containing "content", "buffered_content", "function_calls",
            "is_final", and "raw_chunk".
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

    def astream_completion(
        self,
        response: Any,
        agent: Any | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield structured chunks from a streaming Ollama response asynchronously.

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

        return _gen()

    def parse_tool_calls(self, raw_data: Any) -> list[dict[str, Any]]:
        """Extract tool calls from an Ollama API response.

        The default implementation returns an empty list since Ollama's API
        does not natively support structured tool calls.

        Args:
            raw_data: The raw response object.

        Returns:
            An empty list.
        """
        return []

    def fetch_model_info(self) -> dict[str, Any]:
        """Return model metadata by querying the Ollama API's /api/show endpoint.

        Returns:
            A dict with "max_model_len", "parameter_size", "family",
            and "quantization_level" if available.
        """

        try:
            with httpx.Client(base_url=self.config.base_url or "", timeout=10.0) as client:
                resp = client.post("/api/show", json={"name": self.config.model})
                if resp.status_code == 200:
                    data = resp.json()
                    model_info = data.get("model_info", {})
                    details = data.get("details", {})
                    context_len = (
                        model_info.get("context_length")
                        or model_info.get("llama.context_length")
                        or model_info.get("num_ctx")
                    )
                    return {
                        "max_model_len": context_len,
                        "parameter_size": details.get("parameter_size"),
                        "family": details.get("family"),
                        "quantization_level": details.get("quantization_level"),
                    }
        except Exception:
            pass
        return {}

    async def close(self) -> None:
        """Close the HTTP client and release resources."""

        if self.client:
            await self.client.aclose()


class LocalLLM(OllamaLLM):
    """Local llm.

    Inherits from: OllamaLLM
    """

    pass
