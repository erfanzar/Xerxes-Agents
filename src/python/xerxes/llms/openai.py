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
"""Openai module for Xerxes.

Exports:
    - OpenAILLM"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any
from urllib.parse import urlparse

from .base import BaseLLM, LLMConfig


class _AsyncIteratorFromSyncStream:
    """Async iterator from sync stream."""

    def __init__(self, iterator: Any):
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            iterator (Any): IN: iterator. OUT: Consumed during execution.
        Returns:
            Any: OUT: Result of the operation."""

        self._iterator = iter(iterator)
        self._sentinel = object()

    def __aiter__(self) -> _AsyncIteratorFromSyncStream:
        """Dunder method for aiter.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            _AsyncIteratorFromSyncStream: OUT: Result of the operation."""

        return self

    async def __anext__(self) -> Any:
        """Asynchronously Dunder method for anext.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            Any: OUT: Result of the operation."""

        value = await asyncio.to_thread(next, self._iterator, self._sentinel)
        if value is self._sentinel:
            raise StopAsyncIteration
        return value


class OpenAILLM(BaseLLM):
    """Open aillm.

    Inherits from: BaseLLM
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        client: Any | None = None,
        async_client: Any | None = None,
        **kwargs,
    ):
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            config (LLMConfig | None, optional): IN: config. Defaults to None. OUT: Consumed during execution.
            client (Any | None, optional): IN: client. Defaults to None. OUT: Consumed during execution.
            async_client (Any | None, optional): IN: async client. Defaults to None. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            Any: OUT: Result of the operation."""

        if config is None:
            config = LLMConfig(
                model=kwargs.pop("model", "gpt-4o-mini"),
                api_key=kwargs.pop("api_key", None),
                base_url=kwargs.pop("base_url", None),
                **kwargs,
            )

        self.client = client
        self.async_client = async_client
        super().__init__(config)

    def _initialize_client(self) -> None:
        """Internal helper to initialize client.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        if self.client is None:
            try:
                from openai import AsyncOpenAI, OpenAI
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
            if self.async_client is None:
                self.async_client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout,
                )

        self._auto_fetch_model_info()

    def _supports_openai_compatible_sampling_params(self) -> bool:
        """Internal helper to supports openai compatible sampling params.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            bool: OUT: Result of the operation."""

        if not self.config.base_url:
            return False

        hostname = (urlparse(self.config.base_url).hostname or "").lower()
        if not hostname:
            return True

        official_hosts = {
            "api.openai.com",
            "openai.azure.com",
        }
        return hostname not in official_hosts and not hostname.endswith(".openai.azure.com")

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
            tools (list[dict] | None, optional): IN: tools. Defaults to None. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            Any: OUT: Result of the operation."""

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        compat_top_k = kwargs.pop("top_k", None)
        if compat_top_k is None:
            compat_top_k = self.config.top_k

        compat_min_p = kwargs.pop("min_p", None)
        if compat_min_p is None:
            compat_min_p = self.config.min_p

        compat_repetition_penalty = kwargs.pop("repetition_penalty", None)
        if compat_repetition_penalty is None:
            compat_repetition_penalty = self.config.repetition_penalty

        request_extra_body = kwargs.pop("extra_body", None)
        config_extra_body = self.config.extra_params.get("extra_body", {})
        merged_extra_body = {}
        if isinstance(config_extra_body, dict):
            merged_extra_body.update(config_extra_body)
        if isinstance(request_extra_body, dict):
            merged_extra_body.update(request_extra_body)

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

        if self._supports_openai_compatible_sampling_params():
            if compat_top_k is not None:
                merged_extra_body["top_k"] = compat_top_k
            if compat_min_p is not None:
                merged_extra_body["min_p"] = compat_min_p
            if compat_repetition_penalty is not None:
                merged_extra_body["repetition_penalty"] = compat_repetition_penalty

        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        params.update(filtered_kwargs)

        config_extra_params = {k: v for k, v in self.config.extra_params.items() if k != "extra_body"}
        params.update(config_extra_params)

        if merged_extra_body:
            params["extra_body"] = merged_extra_body

        if self.async_client is not None:
            return await self.async_client.chat.completions.create(**params)

        assert self.client is not None
        response = await asyncio.to_thread(self.client.chat.completions.create, **params)
        if params["stream"] and not hasattr(response, "__aiter__"):
            return _AsyncIteratorFromSyncStream(response)
        return response

    @staticmethod
    def _get_openai_field(obj: Any, field: str) -> Any:
        """Internal helper to get openai field.

        Args:
            obj (Any): IN: obj. OUT: Consumed during execution.
            field (str): IN: field. OUT: Consumed during execution.
        Returns:
            Any: OUT: Result of the operation."""

        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(field)

        value = getattr(obj, field, None)
        if value is not None:
            return value

        model_extra = getattr(obj, "model_extra", None)
        if isinstance(model_extra, dict):
            return model_extra.get(field)
        return None

    @classmethod
    def _stringify_reasoning(cls, value: Any) -> str:
        """Internal helper to stringify reasoning.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            value (Any): IN: value. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list | tuple):
            return "".join(part for item in value if (part := cls._stringify_reasoning(item)))
        if isinstance(value, dict):
            for key in (
                "text",
                "content",
                "summary_text",
                "summary",
                "reasoning_text",
                "reasoning_content",
                "reasoning",
                "delta_reasoning",
                "delta",
                "value",
            ):
                if key in value:
                    text = cls._stringify_reasoning(value[key])
                    if text:
                        return text
            return "".join(part for item in value.values() if (part := cls._stringify_reasoning(item)))

        for key in (
            "text",
            "content",
            "summary_text",
            "summary",
            "reasoning_text",
            "reasoning_content",
            "reasoning",
            "delta_reasoning",
            "delta",
            "value",
        ):
            nested = cls._get_openai_field(value, key)
            if nested is not None and nested is not value:
                text = cls._stringify_reasoning(nested)
                if text:
                    return text

        return ""

    @classmethod
    def _extract_reasoning_from_message(cls, message: Any) -> str:
        """Internal helper to extract reasoning from message.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            message (Any): IN: message. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        for field in ("reasoning_content", "reasoning", "delta_reasoning"):
            text = cls._stringify_reasoning(cls._get_openai_field(message, field))
            if text:
                return text

        content = cls._get_openai_field(message, "content")
        if isinstance(content, list | tuple):
            parts: list[str] = []
            for item in content:
                item_type = str(cls._get_openai_field(item, "type") or "").lower()
                if "reasoning" in item_type:
                    text = cls._stringify_reasoning(item)
                    if text:
                        parts.append(text)
            return "".join(parts)

        return ""

    @classmethod
    def _extract_reasoning_from_chunk(cls, chunk: Any) -> str:
        """Internal helper to extract reasoning from chunk.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            chunk (Any): IN: chunk. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        event_type = str(cls._get_openai_field(chunk, "type") or "")
        if event_type in {
            "response.reasoning.delta",
            "response.reasoning_text.delta",
            "response.reasoning_summary.delta",
            "response.reasoning_summary_text.delta",
        }:
            return cls._stringify_reasoning(cls._get_openai_field(chunk, "delta"))

        choices = cls._get_openai_field(chunk, "choices")
        if choices:
            choice0 = choices[0]
            delta = cls._get_openai_field(choice0, "delta")
            if delta is not None:
                for field in ("reasoning_content", "reasoning", "delta_reasoning"):
                    text = cls._stringify_reasoning(cls._get_openai_field(delta, field))
                    if text:
                        return text

        delta_reasoning = cls._get_openai_field(chunk, "delta_reasoning")
        if delta_reasoning is not None:
            return cls._stringify_reasoning(delta_reasoning)

        return ""

    def extract_content(self, response: Any) -> str:
        """Extract content.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            response (Any): IN: response. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message

            if hasattr(message, "content") and message.content:
                return message.content

            if hasattr(message, "tool_calls") and message.tool_calls:
                return ""

        return ""

    def extract_reasoning_content(self, response: Any) -> str:
        """Extract reasoning content.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            response (Any): IN: response. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            reasoning = self._extract_reasoning_from_message(message)
            if reasoning:
                return reasoning
        return ""

    async def process_streaming_response(self, response: Any, callback: Callable[[str, Any], None]) -> str:
        """Asynchronously Process streaming response.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            response (Any): IN: response. OUT: Consumed during execution.
            callback (Callable[[str, Any], None]): IN: callback. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        accumulated_content = ""

        for chunk in response:
            reasoning = self._extract_reasoning_from_chunk(chunk)
            if reasoning:
                callback(reasoning, chunk)

            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    content = delta.content
                    accumulated_content += content
                    callback(content, chunk)

        return accumulated_content

    def stream_completion(self, response: Any, agent: Any | None = None) -> Iterator[dict[str, Any]]:
        """Stream completion.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            response (Any): IN: response. OUT: Consumed during execution.
            agent (Any | None, optional): IN: agent. Defaults to None. OUT: Consumed during execution.
        Returns:
            Iterator[dict[str, Any]]: OUT: Result of the operation."""

        buffered_content = ""
        buffered_reasoning_content = ""
        function_calls = []
        tool_call_accumulator: dict[int, dict[str, Any]] = {}

        for chunk in response:
            chunk_data = {
                "content": None,
                "buffered_content": buffered_content,
                "reasoning_content": None,
                "buffered_reasoning_content": buffered_reasoning_content,
                "function_calls": [],
                "tool_calls": None,
                "streaming_tool_calls": None,
                "raw_chunk": chunk,
                "is_final": False,
            }

            reasoning = self._extract_reasoning_from_chunk(chunk)
            if reasoning:
                buffered_reasoning_content += reasoning
                chunk_data["reasoning_content"] = reasoning
                chunk_data["buffered_reasoning_content"] = buffered_reasoning_content

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
            buffered_reasoning_content = ""
            function_calls: list[dict[str, Any]] = []
            tool_call_accumulator: dict[int, dict[str, Any]] = {}

            async for chunk in response:
                chunk_data = {
                    "content": None,
                    "buffered_content": buffered_content,
                    "reasoning_content": None,
                    "buffered_reasoning_content": buffered_reasoning_content,
                    "function_calls": [],
                    "tool_calls": None,
                    "streaming_tool_calls": None,
                    "raw_chunk": chunk,
                    "is_final": False,
                }

                reasoning = self._extract_reasoning_from_chunk(chunk)
                if reasoning:
                    buffered_reasoning_content += reasoning
                    chunk_data["reasoning_content"] = reasoning
                    chunk_data["buffered_reasoning_content"] = buffered_reasoning_content

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

        return _gen()

    def parse_tool_calls(self, raw_data: Any) -> list[dict[str, Any]]:
        """Parse tool calls.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            raw_data (Any): IN: raw data. OUT: Consumed during execution.
        Returns:
            list[dict[str, Any]]: OUT: Result of the operation."""

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

    def fetch_model_info(self) -> dict[str, Any]:
        """Fetch model info.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

        try:
            assert self.client is not None
            models = self.client.models.list()
            for model in models.data:
                if model.id == self.config.model:
                    return {
                        "max_model_len": getattr(model, "max_model_len", None),
                        "metadata": getattr(model, "metadata", {}),
                    }
        except Exception:
            pass
        return {}

    async def close(self) -> None:
        """Asynchronously Close.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        if self.async_client is not None:
            if hasattr(self.async_client, "aclose"):
                await self.async_client.aclose()
            elif hasattr(self.async_client, "close"):
                maybe_result = self.async_client.close()
                if hasattr(maybe_result, "__await__"):
                    await maybe_result

        if self.client is not None:
            if hasattr(self.client, "close"):
                maybe_result = self.client.close()
                if hasattr(maybe_result, "__await__"):
                    await maybe_result
