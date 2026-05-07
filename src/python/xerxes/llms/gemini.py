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
"""Gemini module for Xerxes.

Exports:
    - GeminiLLM"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from .base import BaseLLM, LLMConfig


class GeminiLLM(BaseLLM):
    """Gemini LLM client using the Google GenerativeAI SDK.

    Wraps the Google GenerativeAI API to provide completion and streaming
    support for Gemini models.

    Attributes:
        client: The underlying GenerativeModel instance.
        genai: The google.generativeai module.
    """

    def __init__(self, config: LLMConfig | None = None, client: Any | None = None, **kwargs):
        """Initialize the Gemini LLM client.

        Args:
            config: Optional LLM configuration. If None, defaults to "gemini-pro".
            client: Optional pre-configured GenerativeModel instance.
            **kwargs: Additional configuration fields forwarded to LLMConfig.
        """

        if config is None:
            config = LLMConfig(model=kwargs.pop("model", "gemini-pro"), api_key=kwargs.pop("api_key", None), **kwargs)

        self.client = client
        self.genai: Any = None
        super().__init__(config)

    def _initialize_client(self) -> None:
        """Initialize the Google GenerativeAI client and configure authentication.

        Imports google.generativeai, reads the API key from environment or
        config, and creates a GenerativeModel instance.
        """

        try:
            import google.generativeai as genai

            self.genai = genai
        except ImportError as e:
            raise ImportError(
                "Google GenerativeAI library not installed. Install with: pip install google-generativeai"
            ) from e

        api_key = self.config.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not provided")

        self.genai.configure(api_key=api_key)

        if self.client is None:
            self.client = self.genai.GenerativeModel(self.config.model)

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
        """Generate a completion response from the Gemini API.

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
            A dict response from the API, or a streaming iterator if stream=True.
        """

        if model and model != self.config.model:
            client = self.genai.GenerativeModel(model)
        else:
            client = self.client

        if isinstance(prompt, list):
            content = self._format_messages_for_gemini(prompt)
        else:
            content = prompt

        generation_config = self.genai.GenerationConfig(
            temperature=temperature if temperature is not None else self.config.temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
            top_p=top_p if top_p is not None else self.config.top_p,
        )

        if stop or self.config.stop:
            generation_config.stop_sequences = stop or self.config.stop

        if self.config.top_k:
            generation_config.top_k = self.config.top_k

        use_stream = stream if stream is not None else self.config.stream

        try:
            if use_stream:
                return client.generate_content(content, generation_config=generation_config, stream=True, **kwargs)
            else:
                response = client.generate_content(content, generation_config=generation_config, stream=False, **kwargs)
                return response
        except Exception as e:
            raise RuntimeError(f"Gemini API request failed: {e}") from e

    def _format_messages_for_gemini(self, messages: list[dict[str, str]]) -> str:
        """Convert OpenAI-style message list to a Gemini-formatted string.

        Args:
            messages: List of message dicts with role and content.

        Returns:
            A formatted string with "Role: content" sections joined by newlines.
        """

        formatted_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
            else:
                formatted_parts.append(content)

        return "\n\n".join(formatted_parts)

    def extract_content(self, response: Any) -> str:
        """Extract text content from a Gemini API response.

        Args:
            response: The raw API response object.

        Returns:
            The extracted text from the response.
        """

        if hasattr(response, "text"):
            return response.text
        elif hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                parts = candidate.content.parts
                if parts:
                    return parts[0].text
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

        for chunk in response:
            if hasattr(chunk, "text"):
                content = chunk.text
                accumulated_content += content
                callback(content, chunk)
            elif hasattr(chunk, "candidates") and chunk.candidates:
                candidate = chunk.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    parts = candidate.content.parts
                    if parts:
                        content = parts[0].text
                        accumulated_content += content
                        callback(content, chunk)

        return accumulated_content

    def stream_completion(
        self,
        response: Any,
        agent: Any | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield structured chunks from a streaming Gemini response.

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

            if hasattr(chunk, "text") and chunk.text:
                buffered_content += chunk.text
                chunk_data["content"] = chunk.text
                chunk_data["buffered_content"] = buffered_content
            elif hasattr(chunk, "candidates") and chunk.candidates:
                candidate = chunk.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    parts = candidate.content.parts
                    if parts:
                        text = parts[0].text
                        buffered_content += text
                        chunk_data["content"] = text
                        chunk_data["buffered_content"] = buffered_content

            yield chunk_data

        yield {
            "content": None,
            "buffered_content": buffered_content,
            "function_calls": [],
            "tool_calls": None,
            "raw_chunk": None,
            "is_final": True,
        }

    def astream_completion(
        self,
        response: Any,
        agent: Any | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield structured chunks from a streaming Gemini response asynchronously.

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

                if hasattr(chunk, "text") and chunk.text:
                    buffered_content += chunk.text
                    chunk_data["content"] = chunk.text
                    chunk_data["buffered_content"] = buffered_content
                elif hasattr(chunk, "candidates") and chunk.candidates:
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                        parts = candidate.content.parts
                        if parts:
                            text = parts[0].text
                            buffered_content += text
                            chunk_data["content"] = text
                            chunk_data["buffered_content"] = buffered_content

                yield chunk_data

            yield {
                "content": None,
                "buffered_content": buffered_content,
                "function_calls": [],
                "tool_calls": None,
                "raw_chunk": None,
                "is_final": True,
            }

        return _gen()

    def parse_tool_calls(self, raw_data: Any) -> list[dict[str, Any]]:
        """Extract tool calls from a Gemini API response.

        Args:
            raw_data: The raw response object containing function call parts.

        Returns:
            A list of tool call dicts with "id", "name", and "arguments" keys.
        """

        tool_calls = []
        if hasattr(raw_data, "candidates") and raw_data.candidates:
            for candidate in raw_data.candidates:
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call"):
                            fc = part.function_call
                            import json as _json

                            args = getattr(fc, "args", None)
                            if args is not None:
                                try:
                                    args_str = _json.dumps(dict(args))
                                except (TypeError, ValueError):
                                    args_str = str(args)
                            else:
                                args_str = "{}"
                            tool_calls.append(
                                {
                                    "id": getattr(fc, "id", None),
                                    "name": fc.name,
                                    "arguments": args_str,
                                }
                            )
        return tool_calls

    def fetch_model_info(self) -> dict[str, Any]:
        """Return model metadata from the Gemini API.

        Returns:
            A dict with "max_model_len" and "output_token_limit" if available.
        """

        try:
            model_info = self.genai.get_model(f"models/{self.config.model}")
            return {
                "max_model_len": getattr(model_info, "input_token_limit", None),
                "output_token_limit": getattr(model_info, "output_token_limit", None),
            }
        except Exception:
            pass
        return {}
