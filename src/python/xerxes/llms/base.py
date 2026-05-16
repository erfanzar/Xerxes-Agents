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
"""Shared :class:`LLMConfig` and :class:`BaseLLM` interface.

Every provider adapter implements this interface so the rest of
Xerxes can stream completions, parse tool calls, and manage the
underlying client lifecycle without caring which vendor backs the
model. Sampling defaults live on :class:`LLMConfig`; per-call
overrides pass through ``generate_completion`` kwargs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMConfig:
    """Configuration for LLM client interactions.

    Encapsulates all parameters needed to initialize and control an LLM
    client, including model selection, sampling parameters, authentication,
    and networking options.

    Attributes:
        model: The model identifier (e.g., "gpt-4o").
        temperature: Sampling temperature controlling randomness.
        max_tokens: Maximum tokens to generate in a response.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling parameter.
        min_p: Minimum probability threshold for sampling.
        frequency_penalty: Penalty for token frequency in responses.
        presence_penalty: Penalty for token presence in responses.
        repetition_penalty: Repetition penalty multiplier.
        stop: List of stop sequences to halt generation.
        stream: Whether streaming responses are enabled by default.
        api_key: API key for the provider (or resolved from environment).
        base_url: Base URL for the provider's API endpoint.
        timeout: Request timeout in seconds.
        retry_attempts: Number of retries on transient failures.
        extra_params: Additional provider-specific parameters.
        max_model_len: Maximum context window length for the model.
        model_metadata: Additional metadata about the model.
    """

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
    """Abstract base class for LLM client implementations.

    Defines the interface that all provider-specific LLM clients must implement,
    including synchronous and asynchronous completion, streaming, tool call parsing,
    and lifecycle management. Subclasses handle provider-specific HTTP clients,
    authentication, and response parsing.

    Attributes:
        config: The LLM configuration used by this client.
    """

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize the LLM client with the given configuration.

        Args:
            config: Optional LLM configuration. If None, creates a default config
                with model="default" and any provided kwargs.
            **kwargs: Additional configuration fields passed to LLMConfig if
                config is not provided.
        """
        self.config = config or LLMConfig(model="default", **kwargs)
        self._initialize_client()

    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialize the provider-specific HTTP client.

        Subclasses should create and configure their client instance here,
        including setting authentication headers and timeouts.
        """
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
        """Generate a completion response from the LLM.

        Args:
            prompt: A string prompt or a list of message dicts in OpenAI format.
            model: Override the default model for this request.
            temperature: Override sampling temperature for this request.
            max_tokens: Override maximum tokens to generate.
            top_p: Override nucleus sampling threshold.
            stop: Override stop sequences.
            stream: Whether to stream the response.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The raw response from the provider (a dict or streaming iterator,
            depending on the provider and stream flag).
        """
        pass

    @abstractmethod
    def extract_content(self, response: Any) -> str:
        """Extract the text content from an LLM response.

        Args:
            response: The raw response from generate_completion.

        Returns:
            The extracted text string from the response.
        """
        pass

    @abstractmethod
    async def process_streaming_response(
        self,
        response: Any,
        callback: Callable[[str, Any], None],
    ) -> str:
        """Process a streaming response, invoking a callback for each chunk.

        Args:
            response: The streaming response iterator.
            callback: A callable invoked with (text_chunk, raw_chunk) for each
                streamed text segment.

        Returns:
            The concatenated text of all streamed chunks.
        """
        pass

    @abstractmethod
    def stream_completion(
        self,
        response: Any,
        agent: Any | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Translate a provider stream into normalised chunk dicts.

        Each yielded dict carries at minimum ``content`` (the new text
        delta), ``buffered_content`` (text-so-far), ``function_calls``
        / ``tool_calls`` (typed tool call lists when present), and
        ``is_final`` (``True`` on the terminating chunk). Provider
        adapters may attach extra keys (e.g. ``usage``, ``thinking``).
        """
        pass

    @abstractmethod
    def astream_completion(
        self,
        response: Any,
        agent: Any | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async variant of :meth:`stream_completion` with the same chunk shape."""
        pass

    def parse_tool_calls(self, raw_data: Any) -> list[dict[str, Any]]:
        """Extract structured tool calls from a response.

        The default implementation returns an empty list. Subclasses should
        override this if the provider supports function/tool calling.

        Args:
            raw_data: The raw response or parsed response object.

        Returns:
            A list of tool call dicts with "id", "name", and "arguments" keys.
        """
        return []

    def validate_config(self) -> None:
        """Validate the current configuration values.

        Checks that model, temperature, max_tokens, and top_p are within
        acceptable ranges.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if not self.config.model:
            raise ValueError("Model name is required")

        if self.config.temperature < 0 or self.config.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")

        if self.config.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if self.config.top_p <= 0 or self.config.top_p > 1:
            raise ValueError("top_p must be between 0 and 1")

    async def __aenter__(self):
        """Return ``self`` so ``async with`` blocks bind the client."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close the underlying HTTP client on context exit."""
        await self.close()

    @abstractmethod
    async def close(self) -> None:
        """Close the HTTP client and release resources.

        Should be called when the LLM client is no longer needed, either
        explicitly or via the async context manager protocol.
        """
        pass

    def format_messages(self, messages: list[dict[str, str]], system_prompt: str | None = None) -> list[dict[str, str]]:
        """Prepend a system prompt to a message list.

        Args:
            messages: The list of message dicts to format.
            system_prompt: Optional system prompt to prepend.

        Returns:
            The messages with system prompt inserted at the front if provided.
        """
        formatted = []

        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})

        formatted.extend(messages)
        return formatted

    def fetch_model_info(self) -> dict[str, Any]:
        """Retrieve metadata about the configured model.

        Subclasses can override this to query the provider for model-specific
        information such as context limits.

        Returns:
            A dictionary with model metadata fields (e.g., "max_model_len").
        """
        return {}

    def _auto_fetch_model_info(self) -> None:
        """Fetch model info and populate the configuration with the results.

        Called automatically during initialization to set max_model_len and
        other metadata if available.
        """
        try:
            info = self.fetch_model_info()
            if info.get("max_model_len"):
                self.config.max_model_len = info["max_model_len"]
            self.config.model_metadata.update(info)
        except Exception:
            pass

    def get_model_info(self) -> dict[str, Any]:
        """Return a summary of the current model configuration.

        Returns:
            A dictionary containing provider, model name, temperature, max_tokens,
            max_model_len, and stream settings.
        """
        return {
            "provider": self.__class__.__name__.replace("LLM", ""),
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "max_model_len": self.config.max_model_len,
            "stream": self.config.stream,
        }

    def __repr__(self) -> str:
        """Return ``Provider(model='...', temperature=N)``."""
        info = self.get_model_info()
        return f"{info['provider']}(model='{info['model']}', temperature={info['temperature']})"
