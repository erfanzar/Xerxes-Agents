
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

"""Google Gemini LLM implementation."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from .base import BaseLLM, LLMConfig


class GeminiLLM(BaseLLM):
    """Google Gemini LLM implementation."""

    def __init__(self, config: LLMConfig | None = None, client: Any | None = None, **kwargs):
        """Initialize Gemini LLM.

        Args:
            config: LLM configuration
            client: Optional Gemini client instance
            **kwargs: Additional configuration parameters
        """

        if config is None:
            config = LLMConfig(model=kwargs.pop("model", "gemini-pro"), api_key=kwargs.pop("api_key", None), **kwargs)

        self.client = client
        self.genai = None
        super().__init__(config)

    def _initialize_client(self) -> None:
        """Initialize the Gemini client if not provided."""
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
        """Generate completion using Google Gemini API.

        Args:
            prompt: Text prompt or list of messages
            model: Model override
            temperature: Temperature override
            max_tokens: Max tokens override
            top_p: Top-p override
            stop: Stop sequences override
            stream: Stream override
            **kwargs: Additional Gemini-specific parameters

        Returns:
            Gemini completion response
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
        """Format messages for Gemini.

        Gemini expects a simple string or structured content.
        For now, we'll concatenate messages into a string.

        Args:
            messages: List of message dictionaries

        Returns:
            Formatted prompt string
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
        """Extract content from Gemini response.

        Args:
            response: Gemini GenerateContentResponse

        Returns:
            The text content from the response
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
        """Process streaming response from Gemini.

        Args:
            response: Gemini streaming response
            callback: Callback for each chunk

        Returns:
            Complete accumulated content
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
        """Stream completion chunks from Gemini.

        Args:
            response: Gemini streaming response
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

            try:
                chunk_data["is_final"] = False
            except StopIteration:
                chunk_data["is_final"] = True

            yield chunk_data

    async def astream_completion(
        self,
        response: Any,
        agent: Any | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async stream completion chunks from Gemini.

        Args:
            response: Gemini async streaming response
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

    def parse_tool_calls(self, raw_data: Any) -> list[dict[str, Any]]:
        """Parse tool/function calls from Gemini format.

        Args:
            raw_data: Gemini function call data

        Returns:
            Standardized list of tool calls
        """

        tool_calls = []
        if hasattr(raw_data, "candidates") and raw_data.candidates:
            for candidate in raw_data.candidates:
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call"):
                            fc = part.function_call
                            tool_calls.append(
                                {
                                    "id": getattr(fc, "id", None),
                                    "name": fc.name,
                                    "arguments": str(fc.args) if hasattr(fc, "args") else "",
                                }
                            )
        return tool_calls
