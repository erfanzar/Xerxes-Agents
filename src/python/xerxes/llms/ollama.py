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
"""Ollama module for Xerxes.

Exports:
    - OllamaLLM
    - LocalLLM"""

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
    """Ollama llm.

    Inherits from: BaseLLM
    """

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            config (LLMConfig | None, optional): IN: config. Defaults to None. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            Any: OUT: Result of the operation."""

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
        """Internal helper to initialize client.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

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
        """Asynchronously Internal helper to stream completion.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            endpoint (str): IN: endpoint. OUT: Consumed during execution.
            payload (dict): IN: payload. OUT: Consumed during execution.
        Returns:
            AsyncIterator[dict]: OUT: Result of the operation."""

        assert self.client is not None
        async with self.client.stream("POST", endpoint, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    yield json.loads(line)

    def extract_content(self, response: Any) -> str:
        """Extract content.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            response (Any): IN: response. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

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
        """Asynchronously Process streaming response.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            response (Any): IN: response. OUT: Consumed during execution.
            callback (Callable[[str, Any], None]): IN: callback. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

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
        """Stream completion.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            response (Any): IN: response. OUT: Consumed during execution.
            agent (Any | None, optional): IN: agent. Defaults to None. OUT: Consumed during execution.
        Returns:
            Iterator[dict[str, Any]]: OUT: Result of the operation."""

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
        """Parse tool calls.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            raw_data (Any): IN: raw data. OUT: Consumed during execution.
        Returns:
            list[dict[str, Any]]: OUT: Result of the operation."""

        return []

    def fetch_model_info(self) -> dict[str, Any]:
        """Fetch model info.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

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
        """Asynchronously Close.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        if self.client:
            await self.client.aclose()


class LocalLLM(OllamaLLM):
    """Local llm.

    Inherits from: OllamaLLM
    """

    pass
