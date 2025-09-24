
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

"""Base LLM interface for all providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""

    model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    top_k: int | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: list[str] | None = None
    stream: bool = False
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 60.0
    retry_attempts: int = 3
    extra_params: dict[str, Any] = field(default_factory=dict)


class BaseLLM(ABC):
    """Base class for all LLM providers."""

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize the LLM provider.

        Args:
            config: LLM configuration object
            **kwargs: Additional provider-specific arguments
        """
        self.config = config or LLMConfig(model="default", **kwargs)
        self._initialize_client()

    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialize the underlying client for the provider."""
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
        """Generate a completion from the LLM.

        Args:
            prompt: The prompt string or list of messages
            model: Model to use (overrides config)
            temperature: Temperature for sampling (overrides config)
            max_tokens: Maximum tokens to generate (overrides config)
            top_p: Top-p sampling parameter (overrides config)
            stop: Stop sequences (overrides config)
            stream: Whether to stream the response (overrides config)
            **kwargs: Additional provider-specific parameters

        Returns:
            The completion response (format varies by provider)
        """
        pass

    @abstractmethod
    def extract_content(self, response: Any) -> str:
        """Extract text content from provider response.

        Args:
            response: The raw response from the provider

        Returns:
            The extracted text content
        """
        pass

    @abstractmethod
    async def process_streaming_response(
        self,
        response: Any,
        callback: Callable[[str, Any], None],
    ) -> str:
        """Process a streaming response from the provider.

        Args:
            response: The streaming response object
            callback: Function to call for each chunk (content, raw_chunk)

        Returns:
            The complete accumulated content
        """
        pass

    @abstractmethod
    def stream_completion(
        self,
        response: Any,
        agent: Any | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream completion chunks with function call detection.

        Args:
            response: The streaming response from the provider
            agent: Optional agent for function detection

        Yields:
            Dictionary with streaming chunk information:
            - content: Text content in this chunk
            - buffered_content: Accumulated content so far
            - function_calls: List of detected function calls
            - tool_calls: Raw tool call data from provider
            - is_final: Whether this is the final chunk
        """
        pass

    @abstractmethod
    async def astream_completion(
        self,
        response: Any,
        agent: Any | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async stream completion chunks with function call detection.

        Args:
            response: The async streaming response from the provider
            agent: Optional agent for function detection

        Yields:
            Dictionary with streaming chunk information
        """
        pass

    def parse_tool_calls(self, raw_data: Any) -> list[dict[str, Any]]:
        """Parse tool/function calls from provider-specific format.

        Args:
            raw_data: Provider-specific tool call data

        Returns:
            Standardized list of tool calls
        """
        return []

    def validate_config(self) -> None:
        """Validate the configuration for the provider."""
        if not self.config.model:
            raise ValueError("Model name is required")

        if self.config.temperature < 0 or self.config.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")

        if self.config.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if self.config.top_p <= 0 or self.config.top_p > 1:
            raise ValueError("top_p must be between 0 and 1")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:  # noqa: B027
        """Close any open connections."""
        pass

    def format_messages(self, messages: list[dict[str, str]], system_prompt: str | None = None) -> list[dict[str, str]]:
        """Format messages for the provider.

        Args:
            messages: List of message dictionaries
            system_prompt: Optional system prompt to prepend

        Returns:
            Formatted messages for the provider
        """
        formatted = []

        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})

        formatted.extend(messages)
        return formatted

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model configuration.

        Returns:
            Dictionary with model information
        """
        return {
            "provider": self.__class__.__name__.replace("LLM", ""),
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": self.config.stream,
        }

    def __repr__(self) -> str:
        """String representation of the LLM."""
        info = self.get_model_info()
        return f"{info['provider']}(model='{info['model']}', temperature={info['temperature']})"
