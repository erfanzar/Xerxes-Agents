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


"""OpenAI LLM implementation."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from .base import BaseLLM, LLMConfig


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation using the OpenAI client."""

    def __init__(self, config: LLMConfig | None = None, client: Any | None = None, **kwargs):
        """Initialize OpenAI LLM.

        Args:
            config: LLM configuration
            client: Optional OpenAI client instance
            **kwargs: Additional configuration parameters
        """

        if config is None:
            config = LLMConfig(
                model=kwargs.pop("model", "gpt-4o-mini"),
                api_key=kwargs.pop("api_key", None),
                base_url=kwargs.pop("base_url", None),
                **kwargs,
            )

        self.client = client
        super().__init__(config)

    def _initialize_client(self) -> None:
        """Initialize the OpenAI client if not provided."""
        if self.client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("OpenAI library not installed. Install with: pip install openai") from e

            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key and not self.config.base_url:
                raise ValueError("OpenAI API key not provided and no base URL specified")

            self.client = OpenAI(
                api_key=api_key,
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
        tools: list[dict] | None = None,
        **kwargs,
    ) -> Any:
        """Generate completion using OpenAI API.

        Args:
            prompt: Text prompt or list of messages
            model: Model override
            temperature: Temperature override
            max_tokens: Max tokens override
            top_p: Top-p override
            stop: Stop sequences override
            stream: Stream override
            tools: List of available tools for function calling
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            OpenAI completion response
        """

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        params = {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "top_p": top_p if top_p is not None else self.config.top_p,
            "stream": stream if stream is not None else self.config.stream,
        }

        if stop or self.config.stop:
            params["stop"] = stop or self.config.stop

        if self.config.frequency_penalty:
            params["frequency_penalty"] = self.config.frequency_penalty

        if self.config.presence_penalty:
            params["presence_penalty"] = self.config.presence_penalty

        if tools:
            params["tools"] = tools

            params["tool_choice"] = "auto"

        openai_unsupported = {"top_k", "min_p", "repetition_penalty"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in openai_unsupported and v is not None}
        params.update(filtered_kwargs)
        params.update(self.config.extra_params)

        if params["stream"]:
            return self.client.chat.completions.create(**params)
        else:
            response = self.client.chat.completions.create(**params)
            return response

    def extract_content(self, response: Any) -> str:
        """Extract content from OpenAI response.

        Args:
            response: OpenAI ChatCompletion response

        Returns:
            The text content from the response
        """
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message

            if hasattr(message, "content") and message.content:
                return message.content

            if hasattr(message, "tool_calls") and message.tool_calls:
                return ""

        return ""

    async def process_streaming_response(self, response: Any, callback: Callable[[str, Any], None]) -> str:
        """Process streaming response from OpenAI.

        Args:
            response: OpenAI streaming response
            callback: Callback for each chunk

        Returns:
            Complete accumulated content
        """
        accumulated_content = ""

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    content = delta.content
                    accumulated_content += content
                    callback(content, chunk)

        return accumulated_content

    def stream_completion(self, response: Any, agent: Any | None = None) -> Iterator[dict[str, Any]]:
        """Stream completion chunks with function call detection.

        Args:
            response: OpenAI streaming response
            agent: Optional agent for function detection

        Yields:
            Dictionary with streaming chunk information
        """
        buffered_content = ""
        function_calls = []
        tool_call_accumulator = {}

        for chunk in response:
            chunk_data = {
                "content": None,
                "buffered_content": buffered_content,
                "function_calls": [],
                "tool_calls": None,
                "streaming_tool_calls": None,
                "raw_chunk": chunk,
                "is_final": False,
            }

            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta

                if hasattr(delta, "content") and delta.content:
                    buffered_content += delta.content
                    chunk_data["content"] = delta.content
                    chunk_data["buffered_content"] = buffered_content

                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    streaming_tool_calls = {}
                    accumulated_tool_calls = {}

                    for tool_call_delta in delta.tool_calls:
                        idx = getattr(tool_call_delta, "index", 0)
                        if isinstance(tool_call_delta, dict):
                            idx = tool_call_delta.get("index", 0)

                        if idx not in tool_call_accumulator:
                            tool_call_accumulator[idx] = {
                                "id": None,
                                "type": "function",
                                "function": {"name": None, "arguments": ""},
                            }

                        streaming_update = {}

                        if hasattr(tool_call_delta, "id") and tool_call_delta.id:
                            tool_call_accumulator[idx]["id"] = tool_call_delta.id
                            streaming_update["id"] = tool_call_delta.id

                        if hasattr(tool_call_delta, "function") and tool_call_delta.function:
                            func = tool_call_delta.function
                            if hasattr(func, "name") and func.name:
                                tool_call_accumulator[idx]["function"]["name"] = func.name
                                streaming_update["name"] = func.name
                            if hasattr(func, "arguments") and func.arguments:
                                tool_call_accumulator[idx]["function"]["arguments"] += func.arguments
                                streaming_update["arguments"] = func.arguments

                        if streaming_update:
                            streaming_tool_calls[idx] = streaming_update

                        accumulated_tool_calls[idx] = {
                            "id": tool_call_accumulator[idx]["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call_accumulator[idx]["function"]["name"],
                                "arguments": tool_call_accumulator[idx]["function"]["arguments"],
                            },
                        }

                    chunk_data["tool_calls"] = accumulated_tool_calls if accumulated_tool_calls else None
                    chunk_data["streaming_tool_calls"] = streaming_tool_calls if streaming_tool_calls else None

                if chunk.choices[0].finish_reason:
                    chunk_data["is_final"] = True

                    if tool_call_accumulator:
                        for idx in sorted(tool_call_accumulator.keys()):
                            tc = tool_call_accumulator[idx]
                            if tc["id"] and tc["function"]["name"]:
                                function_calls.append(
                                    {
                                        "id": tc["id"],
                                        "name": tc["function"]["name"],
                                        "arguments": tc["function"]["arguments"],
                                    }
                                )
                        chunk_data["function_calls"] = function_calls

            yield chunk_data

    async def astream_completion(
        self,
        response: Any,
        agent: Any | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async stream completion chunks with function call detection.

        Args:
            response: OpenAI async streaming response
            agent: Optional agent for function detection

        Yields:
            Dictionary with streaming chunk information
        """
        buffered_content = ""
        function_calls = []
        tool_call_accumulator = {}

        async for chunk in response:
            chunk_data = {
                "content": None,
                "buffered_content": buffered_content,
                "function_calls": [],
                "tool_calls": None,
                "streaming_tool_calls": None,
                "raw_chunk": chunk,
                "is_final": False,
            }

            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta

                if hasattr(delta, "content") and delta.content:
                    buffered_content += delta.content
                    chunk_data["content"] = delta.content
                    chunk_data["buffered_content"] = buffered_content

                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    streaming_tool_calls = {}
                    accumulated_tool_calls = {}

                    for tool_call_delta in delta.tool_calls:
                        idx = getattr(tool_call_delta, "index", 0)
                        if isinstance(tool_call_delta, dict):
                            idx = tool_call_delta.get("index", 0)

                        if idx not in tool_call_accumulator:
                            tool_call_accumulator[idx] = {
                                "id": None,
                                "type": "function",
                                "function": {"name": None, "arguments": ""},
                            }

                        streaming_update = {}

                        if hasattr(tool_call_delta, "id") and tool_call_delta.id:
                            tool_call_accumulator[idx]["id"] = tool_call_delta.id
                            streaming_update["id"] = tool_call_delta.id

                        if hasattr(tool_call_delta, "function") and tool_call_delta.function:
                            func = tool_call_delta.function
                            if hasattr(func, "name") and func.name:
                                tool_call_accumulator[idx]["function"]["name"] = func.name
                                streaming_update["name"] = func.name
                            if hasattr(func, "arguments") and func.arguments:
                                tool_call_accumulator[idx]["function"]["arguments"] += func.arguments
                                streaming_update["arguments"] = func.arguments

                        if streaming_update:
                            streaming_tool_calls[idx] = streaming_update

                        accumulated_tool_calls[idx] = {
                            "id": tool_call_accumulator[idx]["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call_accumulator[idx]["function"]["name"],
                                "arguments": tool_call_accumulator[idx]["function"]["arguments"],
                            },
                        }

                    chunk_data["tool_calls"] = accumulated_tool_calls if accumulated_tool_calls else None
                    chunk_data["streaming_tool_calls"] = streaming_tool_calls if streaming_tool_calls else None

                if chunk.choices[0].finish_reason:
                    chunk_data["is_final"] = True

                    if tool_call_accumulator:
                        for idx in sorted(tool_call_accumulator.keys()):
                            tc = tool_call_accumulator[idx]
                            if tc["id"] and tc["function"]["name"]:
                                function_calls.append(
                                    {
                                        "id": tc["id"],
                                        "name": tc["function"]["name"],
                                        "arguments": tc["function"]["arguments"],
                                    }
                                )
                        chunk_data["function_calls"] = function_calls

            yield chunk_data

    def parse_tool_calls(self, raw_data: Any) -> list[dict[str, Any]]:
        """Parse tool/function calls from OpenAI format.

        Args:
            raw_data: OpenAI tool call data

        Returns:
            Standardized list of tool calls
        """
        tool_calls = []
        if hasattr(raw_data, "tool_calls") and raw_data.tool_calls:
            for tc in raw_data.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                )
        return tool_calls

    async def close(self) -> None:
        """Close the OpenAI client."""
        if hasattr(self.client, "close"):
            self.client.close()
