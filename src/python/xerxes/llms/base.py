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
"""Base module for Xerxes.

Exports:
    - LLMConfig
    - BaseLLM"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMConfig:
    """Llmconfig.

    Attributes:
        model (str): model.
        temperature (float): temperature.
        max_tokens (int): max tokens.
        top_p (float): top p.
        top_k (int | None): top k.
        min_p (float): min p.
        frequency_penalty (float): frequency penalty.
        presence_penalty (float): presence penalty.
        repetition_penalty (float): repetition penalty.
        stop (list[str] | None): stop.
        stream (bool): stream.
        api_key (str | None): api key.
        base_url (str | None): base url.
        timeout (float): timeout.
        retry_attempts (int): retry attempts.
        extra_params (dict[str, Any]): extra params.
        max_model_len (int | None): max model len.
        model_metadata (dict[str, Any]): model metadata."""

    model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    top_k: int | None = None
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stop: list[str] | None = None
    stream: bool = False
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 60.0
    retry_attempts: int = 3
    extra_params: dict[str, Any] = field(default_factory=dict)
    max_model_len: int | None = None
    model_metadata: dict[str, Any] = field(default_factory=dict)


class BaseLLM(ABC):
    """Base llm.

    Inherits from: ABC
    """

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            config (LLMConfig | None, optional): IN: config. Defaults to None. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            Any: OUT: Result of the operation."""

        self.config = config or LLMConfig(model="default", **kwargs)
        self._initialize_client()

    @abstractmethod
    def _initialize_client(self) -> None:
        """Internal helper to initialize client.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        pass

    @abstractmethod
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

        pass

    @abstractmethod
    def extract_content(self, response: Any) -> str:
        """Extract content.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            response (Any): IN: response. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        pass

    @abstractmethod
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

        pass

    @abstractmethod
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

        pass

    @abstractmethod
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

        pass

    def parse_tool_calls(self, raw_data: Any) -> list[dict[str, Any]]:
        """Parse tool calls.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            raw_data (Any): IN: raw data. OUT: Consumed during execution.
        Returns:
            list[dict[str, Any]]: OUT: Result of the operation."""

        return []

    def validate_config(self) -> None:
        """Validate config.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        if not self.config.model:
            raise ValueError("Model name is required")

        if self.config.temperature < 0 or self.config.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")

        if self.config.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if self.config.top_p <= 0 or self.config.top_p > 1:
            raise ValueError("top_p must be between 0 and 1")

    async def __aenter__(self):
        """Asynchronously Dunder method for aenter.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            Any: OUT: Result of the operation."""

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Asynchronously Dunder method for aexit.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            exc_type (Any): IN: exc type. OUT: Consumed during execution.
            exc_val (Any): IN: exc val. OUT: Consumed during execution.
            exc_tb (Any): IN: exc tb. OUT: Consumed during execution.
        Returns:
            Any: OUT: Result of the operation."""

        await self.close()

    @abstractmethod
    async def close(self) -> None:
        """Asynchronously Close.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        pass

    def format_messages(self, messages: list[dict[str, str]], system_prompt: str | None = None) -> list[dict[str, str]]:
        """Format messages.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            messages (list[dict[str, str]]): IN: messages. OUT: Consumed during execution.
            system_prompt (str | None, optional): IN: system prompt. Defaults to None. OUT: Consumed during execution.
        Returns:
            list[dict[str, str]]: OUT: Result of the operation."""

        formatted = []

        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})

        formatted.extend(messages)
        return formatted

    def fetch_model_info(self) -> dict[str, Any]:
        """Fetch model info.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

        return {}

    def _auto_fetch_model_info(self) -> None:
        """Internal helper to auto fetch model info.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        try:
            info = self.fetch_model_info()
            if info.get("max_model_len"):
                self.config.max_model_len = info["max_model_len"]
            self.config.model_metadata.update(info)
        except Exception:
            pass

    def get_model_info(self) -> dict[str, Any]:
        """Retrieve the model info.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

        return {
            "provider": self.__class__.__name__.replace("LLM", ""),
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "max_model_len": self.config.max_model_len,
            "stream": self.config.stream,
        }

    def __repr__(self) -> str:
        """Dunder method for repr.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            str: OUT: Result of the operation."""

        info = self.get_model_info()
        return f"{info['provider']}(model='{info['model']}', temperature={info['temperature']})"
