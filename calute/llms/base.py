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


"""Base LLM interface for all providers.

This module provides the abstract base classes and configuration dataclasses
for integrating Large Language Model (LLM) providers into the Calute framework.
It defines a standardized interface that all provider implementations must follow,
ensuring consistent behavior across different LLM backends.

The module supports features like:
- Synchronous and asynchronous completion generation
- Streaming responses with function call detection
- Provider-specific configuration with sensible defaults
- Automatic model metadata fetching
- Context manager support for resource management
- Tool/function call parsing from various formats

Supported provider implementations (in separate modules):
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude models)
- Google (Gemini models)
- vLLM (local deployment)
- LiteLLM (unified interface)

Typical usage example:
    from calute.llms.openai_llm import OpenAILLM
    from calute.llms.base import LLMConfig

    config = LLMConfig(
        model="gpt-4",
        temperature=0.7,
        max_tokens=2048,
        api_key="your-api-key"
    )

    llm = OpenAILLM(config)
    response = await llm.generate_completion("Hello, world!")
    content = llm.extract_content(response)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMConfig:
    """Configuration dataclass for LLM providers.

    LLMConfig provides a standardized way to configure LLM provider instances
    with common parameters like model selection, sampling settings, and API
    credentials. All provider implementations accept this configuration,
    though some parameters may be provider-specific.

    Attributes:
        model: The model identifier to use (e.g., 'gpt-4', 'claude-3-opus').
        temperature: Controls randomness in sampling. Higher values (0.8-1.0)
            make output more random, lower values (0.1-0.3) more deterministic.
        max_tokens: Maximum number of tokens to generate in the response.
        top_p: Nucleus sampling parameter. Only tokens comprising the top_p
            probability mass are considered for sampling.
        top_k: Top-k sampling parameter. Only the top k most likely tokens
            are considered. Set to None to disable.
        frequency_penalty: Penalizes tokens based on their frequency in the
            text so far, reducing repetition. Range: -2.0 to 2.0.
        presence_penalty: Penalizes tokens that have appeared at all in the
            text so far, encouraging topic diversity. Range: -2.0 to 2.0.
        stop: List of sequences where the model should stop generating.
        stream: Whether to stream the response token by token.
        api_key: API key for the provider. Can also be set via environment.
        base_url: Custom base URL for API requests (useful for proxies or
            self-hosted instances).
        timeout: Request timeout in seconds.
        retry_attempts: Number of retry attempts for failed requests.
        extra_params: Dictionary for provider-specific parameters not covered
            by the standard configuration options.
        max_model_len: Maximum context length supported by the model.
            Auto-populated by fetch_model_info() when available.
        model_metadata: Dictionary storing additional model information
            fetched from the provider API.

    Example:
        config = LLMConfig(
            model="gpt-4-turbo",
            temperature=0.5,
            max_tokens=4096,
            stream=True,
            api_key="sk-..."
        )
    """

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
    # Model metadata (auto-fetched from provider)
    max_model_len: int | None = None
    model_metadata: dict[str, Any] = field(default_factory=dict)


class BaseLLM(ABC):
    """Abstract base class for all LLM provider implementations.

    BaseLLM defines the standard interface that all LLM provider implementations
    must follow within the Calute framework. It provides common functionality
    for configuration management, message formatting, and resource handling,
    while requiring subclasses to implement provider-specific completion logic.

    This class supports both synchronous and asynchronous operations, streaming
    responses with function call detection, and automatic model metadata fetching.
    It is designed to be used as an async context manager for proper resource
    cleanup.

    Subclasses must implement:
        - _initialize_client(): Set up the provider-specific client
        - generate_completion(): Generate completions from prompts
        - extract_content(): Extract text from provider responses
        - process_streaming_response(): Handle streaming with callbacks
        - stream_completion(): Synchronous streaming with function detection
        - astream_completion(): Asynchronous streaming with function detection

    Attributes:
        config: LLMConfig instance containing provider configuration.

    Example:
        class MyProviderLLM(BaseLLM):
            def _initialize_client(self) -> None:
                self.client = MyProviderClient(api_key=self.config.api_key)

            async def generate_completion(self, prompt, **kwargs):
                return await self.client.complete(prompt)

            def extract_content(self, response) -> str:
                return response.text

        # Usage
        async with MyProviderLLM(config) as llm:
            response = await llm.generate_completion("Hello!")
            print(llm.extract_content(response))
    """

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize the LLM provider.

        Creates a new LLM provider instance with the specified configuration.
        If no config is provided, a default configuration is created using
        the keyword arguments.

        Args:
            config: LLM configuration object. If None, a default config is
                created with model="default" and any provided kwargs.
            **kwargs: Additional provider-specific arguments passed to
                LLMConfig when config is None. Common kwargs include:
                model, temperature, max_tokens, api_key.

        Side Effects:
            - Stores config in self.config
            - Calls _initialize_client() to set up the provider client
        """
        self.config = config or LLMConfig(model="default", **kwargs)
        self._initialize_client()

    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialize the underlying client for the provider.

        This abstract method must be implemented by subclasses to set up
        the provider-specific client instance. It is called automatically
        at the end of __init__.

        Implementations should:
            - Create the provider client using self.config settings
            - Set up authentication using api_key from config
            - Optionally call self._auto_fetch_model_info() to populate
              model metadata

        Raises:
            AuthenticationError: If API key is invalid or missing.
            ConnectionError: If unable to connect to provider API.
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
        """Validate the configuration for the provider.

        Checks that all configuration values are within valid ranges.
        This method can be called explicitly or automatically by
        provider implementations during initialization.

        Raises:
            ValueError: If model name is empty or missing.
            ValueError: If temperature is not between 0 and 2.
            ValueError: If max_tokens is not positive.
            ValueError: If top_p is not between 0 and 1.
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
        """Async context manager entry.

        Enables usage of the LLM provider with async with statement
        for automatic resource cleanup.

        Returns:
            self: The LLM provider instance.

        Example:
            async with OpenAILLM(config) as llm:
                response = await llm.generate_completion("Hello")
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit.

        Automatically closes connections when exiting the context.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        await self.close()

    async def close(self) -> None:  # noqa: B027
        """Close any open connections and release resources.

        This method should be called when done using the LLM provider
        to properly clean up resources. It is called automatically
        when using the provider as an async context manager.

        Override in subclasses to implement provider-specific cleanup
        such as closing HTTP sessions or releasing connection pools.
        """
        pass

    def format_messages(self, messages: list[dict[str, str]], system_prompt: str | None = None) -> list[dict[str, str]]:
        """Format messages for the provider.

        Prepares a list of messages for the LLM API call by optionally
        prepending a system prompt. The default implementation simply
        adds a system message at the beginning if provided.

        Override in subclasses for provider-specific message formatting
        requirements (e.g., role name mappings, message structure).

        Args:
            messages: List of message dictionaries, each containing
                'role' and 'content' keys. Roles are typically 'user',
                'assistant', or 'system'.
            system_prompt: Optional system prompt to prepend as the
                first message with role='system'.

        Returns:
            Formatted list of message dictionaries ready for the API.

        Example:
            messages = [{"role": "user", "content": "Hello"}]
            formatted = llm.format_messages(messages, "You are helpful.")
            # Returns:
            # [
            #     {"role": "system", "content": "You are helpful."},
            #     {"role": "user", "content": "Hello"}
            # ]
        """
        formatted = []

        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})

        formatted.extend(messages)
        return formatted

    def fetch_model_info(self) -> dict[str, Any]:
        """Fetch model metadata from provider API.

        Override in subclasses to implement provider-specific fetching
        of model capabilities and limits. This information can be used
        to optimize token usage and prevent context overflow.

        Common metadata fields include:
            - max_model_len: Maximum context window size in tokens
            - context_window: Alias for max_model_len
            - supports_function_calling: Whether model supports tools
            - supports_vision: Whether model can process images
            - input_token_limit: Maximum input tokens
            - output_token_limit: Maximum output tokens

        Returns:
            Dictionary with model metadata. Empty dict if metadata
            cannot be fetched or is not supported by the provider.

        Note:
            This method is called by _auto_fetch_model_info() during
            client initialization. Errors are silently ignored to
            prevent initialization failures.
        """
        return {}

    def _auto_fetch_model_info(self) -> None:
        """Auto-fetch model metadata and store in config.

        This method should be called at the end of _initialize_client()
        in subclasses to automatically populate model metadata. It calls
        fetch_model_info() and stores the results in the config object.

        The method silently catches and ignores all exceptions to prevent
        initialization failures due to metadata fetching issues (e.g.,
        network errors, unsupported endpoints).

        Side Effects:
            - Sets self.config.max_model_len if available in fetched info
            - Updates self.config.model_metadata with all fetched data
        """
        try:
            info = self.fetch_model_info()
            if info.get("max_model_len"):
                self.config.max_model_len = info["max_model_len"]
            self.config.model_metadata.update(info)
        except Exception:
            pass

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model configuration.

        Returns a dictionary containing the current configuration and
        provider information. Useful for debugging, logging, and
        displaying model status in UIs.

        Returns:
            Dictionary with model information containing:
                - provider: Provider name derived from class name
                - model: Model identifier from config
                - temperature: Current temperature setting
                - max_tokens: Maximum tokens for generation
                - max_model_len: Maximum context length (if known)
                - stream: Whether streaming is enabled
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
        """String representation of the LLM.

        Returns a human-readable string showing the provider name,
        model, and temperature setting for quick identification.

        Returns:
            String in format: "Provider(model='model-name', temperature=0.7)"
        """
        info = self.get_model_info()
        return f"{info['provider']}(model='{info['model']}', temperature={info['temperature']})"
