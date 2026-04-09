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


"""OpenAI LLM provider implementation.

This module provides the OpenAI-specific implementation of the BaseLLM interface
for integrating OpenAI's GPT models into the Calute framework. It supports all
OpenAI Chat Completion API features including streaming, function calling, and
tool use.

The module supports:
- GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, and GPT-4o models
- Synchronous and asynchronous completion generation
- Streaming responses with real-time function call detection
- Tool/function calling with automatic argument accumulation
- OpenAI-compatible API endpoints (Azure, local proxies, etc.)
- Automatic model metadata fetching from /v1/models endpoint

Key features:
- Automatic API key resolution from environment (OPENAI_API_KEY)
- Support for custom base URLs (useful for Azure OpenAI or proxies)
- Filtering of unsupported parameters (top_k, min_p, repetition_penalty)
- Robust streaming with incremental tool call argument accumulation

Typical usage example:
    from calute.llms.openai import OpenAILLM
    from calute.llms.base import LLMConfig

    # Using configuration object
    config = LLMConfig(
        model="gpt-4o",
        temperature=0.7,
        max_tokens=2048,
        api_key="sk-..."
    )
    llm = OpenAILLM(config)

    # Or using kwargs (defaults to gpt-4o-mini)
    llm = OpenAILLM(model="gpt-4", api_key="sk-...")

    # Generate completion
    response = await llm.generate_completion("Explain quantum computing")
    content = llm.extract_content(response)

    # With streaming
    stream_response = await llm.generate_completion(
        "Write a poem",
        stream=True
    )
    for chunk in llm.stream_completion(stream_response):
        if chunk["content"]:
            print(chunk["content"], end="")
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any
from urllib.parse import urlparse

from .base import BaseLLM, LLMConfig


class _AsyncIteratorFromSyncStream:
    """Adapter that wraps a synchronous streaming iterator as an async iterator.

    This class bridges the gap between OpenAI's synchronous streaming client
    and Calute's async-first interface. It uses ``asyncio.to_thread`` to
    offload blocking ``next()`` calls to a thread pool, preventing the event
    loop from being blocked during streaming.

    This is used internally when the synchronous OpenAI client returns a
    synchronous stream but the caller expects an async iterator (e.g., when
    ``generate_completion`` is awaited with ``stream=True``).

    Attributes:
        _iterator: The underlying synchronous iterator being adapted.
        _sentinel: A unique sentinel object used to detect iterator exhaustion
            without catching ``StopIteration`` across threads.

    Example:
        sync_stream = client.chat.completions.create(stream=True, ...)
        async_stream = _AsyncIteratorFromSyncStream(sync_stream)
        async for chunk in async_stream:
            print(chunk)
    """

    def __init__(self, iterator: Any):
        """Initialize the async adapter with a synchronous iterator.

        Args:
            iterator: Any synchronous iterable or iterator to be wrapped.
                Will be converted to an iterator via ``iter()`` if not
                already one.
        """
        self._iterator = iter(iterator)
        self._sentinel = object()

    def __aiter__(self) -> _AsyncIteratorFromSyncStream:
        """Return self as the async iterator.

        Returns:
            This adapter instance, implementing the async iterator protocol.
        """
        return self

    async def __anext__(self) -> Any:
        """Retrieve the next item from the underlying synchronous iterator.

        Offloads the blocking ``next()`` call to a thread pool worker via
        ``asyncio.to_thread``, so the event loop remains responsive during
        synchronous I/O waits.

        Returns:
            The next item from the underlying iterator.

        Raises:
            StopAsyncIteration: When the underlying synchronous iterator
                is exhausted (detected via the sentinel pattern).
        """
        value = await asyncio.to_thread(next, self._iterator, self._sentinel)
        if value is self._sentinel:
            raise StopAsyncIteration
        return value


class OpenAILLM(BaseLLM):
    """OpenAI LLM provider implementation using the official OpenAI Python client.

    OpenAILLM provides a complete implementation of the BaseLLM interface for
    OpenAI's GPT models. It wraps the official OpenAI Python client and handles
    all the complexity of API communication, streaming, and tool call parsing.

    This class supports both standard OpenAI API and OpenAI-compatible endpoints
    (such as Azure OpenAI, LM Studio, or custom proxy servers) through the
    base_url configuration option.

    The implementation automatically:
    - Resolves API keys from environment variables if not provided
    - Filters out unsupported parameters (top_k, min_p, repetition_penalty)
    - Accumulates streaming tool call arguments incrementally
    - Fetches model metadata from the /v1/models endpoint

    Attributes:
        config: LLMConfig instance containing provider configuration.
        client: OpenAI client instance used for API communication.

    Example:
        # Basic usage with environment variable API key
        llm = OpenAILLM(model="gpt-4o")
        response = await llm.generate_completion("Hello!")
        print(llm.extract_content(response))

        # With explicit configuration
        config = LLMConfig(
            model="gpt-4-turbo",
            temperature=0.5,
            max_tokens=4096,
            api_key="sk-..."
        )
        llm = OpenAILLM(config)

        # Using custom base URL (e.g., Azure or local server)
        llm = OpenAILLM(
            model="gpt-4",
            base_url="https://my-resource.openai.azure.com/",
            api_key="azure-key"
        )

        # Inject custom client
        from openai import OpenAI
        custom_client = OpenAI(api_key="sk-...", timeout=120.0)
        llm = OpenAILLM(client=custom_client)
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        client: Any | None = None,
        async_client: Any | None = None,
        **kwargs,
    ):
        """Initialize the OpenAI LLM provider.

        Creates a new OpenAI LLM provider instance with the specified configuration.
        Supports three initialization patterns: explicit config, keyword arguments,
        or injected client instance.

        Args:
            config: LLM configuration object. If None, a default config is created
                using the provided kwargs with model defaulting to "gpt-4o-mini".
            client: Optional pre-configured synchronous OpenAI client instance.
                When provided, the client is used directly without creating a new
                one. Useful for custom authentication or connection pooling scenarios.
            async_client: Optional pre-configured AsyncOpenAI client instance.
                When provided, this async client is used for ``generate_completion``
                calls instead of wrapping the synchronous client with
                ``asyncio.to_thread``. If not provided but ``client`` is also not
                provided, both clients are created automatically.
            **kwargs: Configuration parameters when config is None. Common kwargs:
                - model: Model identifier (default: "gpt-4o-mini")
                - api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
                - base_url: Custom API endpoint URL
                - temperature, max_tokens, top_p: Sampling parameters
                - stream: Enable streaming by default

        Raises:
            ImportError: If the openai package is not installed.
            ValueError: If no API key is provided and no base_url is specified.

        Side Effects:
            - Stores the client in self.client
            - Calls _initialize_client() to set up the OpenAI client
            - Automatically fetches model metadata via _auto_fetch_model_info()

        Example:
            # Using config object
            config = LLMConfig(model="gpt-4", api_key="sk-...")
            llm = OpenAILLM(config)

            # Using kwargs
            llm = OpenAILLM(model="gpt-4o", temperature=0.5)

            # With pre-configured client
            from openai import OpenAI
            client = OpenAI(api_key="sk-...")
            llm = OpenAILLM(client=client)
        """

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
        """Initialize the OpenAI client if not already provided.

        Creates and configures an OpenAI client instance using the configuration
        settings. If a client was already provided during initialization, this
        method skips client creation and only fetches model metadata.

        The method performs the following steps:
        1. Check if a client was already injected
        2. Import the OpenAI library (raises ImportError if not installed)
        3. Resolve API key from config or OPENAI_API_KEY environment variable
        4. Create OpenAI client with api_key, base_url, and timeout settings
        5. Fetch model metadata via _auto_fetch_model_info()

        Raises:
            ImportError: If the openai package is not installed. The error
                message includes installation instructions.
            ValueError: If no API key is provided (via config or environment)
                and no base_url is specified. Base URL allows keyless access
                for local or custom endpoints.

        Side Effects:
            - Sets self.client to a new OpenAI instance (if not already set)
            - Updates self.config.max_model_len and model_metadata via
              _auto_fetch_model_info()
        """
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
        """Check if the configured endpoint supports extra sampling parameters.

        Determines whether the configured ``base_url`` points to a non-official
        OpenAI-compatible endpoint (such as vLLM, LM Studio, or a custom proxy)
        that typically accepts additional sampling parameters like ``top_k``,
        ``min_p``, and ``repetition_penalty`` via ``extra_body``.

        Official OpenAI and Azure OpenAI endpoints do not support these extra
        parameters and will return errors if they are included.

        Returns:
            ``True`` if ``base_url`` is set and does not point to
            ``api.openai.com`` or ``*.openai.azure.com``, indicating a
            third-party endpoint that likely supports extended sampling params.
            ``False`` if no ``base_url`` is configured or the host is an
            official OpenAI endpoint.
        """
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
        """Generate a completion using the OpenAI Chat Completions API.

        Sends a request to OpenAI's chat completions endpoint and returns the
        response. Supports both simple string prompts and full message lists,
        with optional tool/function calling support.

        The method automatically:
        - Converts string prompts to message format
        - Merges config defaults with override parameters
        - Filters out unsupported parameters (top_k, min_p, repetition_penalty)
        - Sets tool_choice to "auto" when tools are provided
        - Applies extra_params from config

        Args:
            prompt: The input to generate completion for. Can be either:
                - A string (converted to a single user message)
                - A list of message dicts with 'role' and 'content' keys
            model: Model identifier override. If None, uses config.model.
            temperature: Sampling temperature override (0.0-2.0). Higher values
                make output more random. If None, uses config.temperature.
            max_tokens: Maximum tokens to generate. If None, uses config.max_tokens.
            top_p: Nucleus sampling parameter (0.0-1.0). If None, uses config.top_p.
            stop: List of stop sequences. When encountered, generation stops.
                If None, uses config.stop.
            stream: Whether to stream the response. If True, returns a streaming
                iterator instead of a complete response. If None, uses config.stream.
            tools: List of tool definitions for function calling. Each tool should
                be a dict with 'type' and 'function' keys following OpenAI's
                tool schema. When provided, tool_choice is set to "auto".
            **kwargs: Additional OpenAI-specific parameters. Unsupported params
                (top_k, min_p, repetition_penalty) are automatically filtered out.
                Examples: response_format, seed, logprobs, user.

        Returns:
            OpenAI ChatCompletion response object. If stream=True, returns a
            streaming iterator that yields ChatCompletionChunk objects.

        Note:
            This method uses the synchronous OpenAI client internally but is
            declared async for interface consistency with other providers.
            For true async, consider using AsyncOpenAI client.

        Example:
            # Simple string prompt
            response = await llm.generate_completion("What is 2+2?")

            # Message list with system prompt
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]
            response = await llm.generate_completion(messages)

            # With tools
            tools = [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {...}
                }
            }]
            response = await llm.generate_completion(prompt, tools=tools)
        """
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

        response = await asyncio.to_thread(self.client.chat.completions.create, **params)
        if params["stream"] and not hasattr(response, "__aiter__"):
            return _AsyncIteratorFromSyncStream(response)
        return response

    @staticmethod
    def _get_openai_field(obj: Any, field: str) -> Any:
        """Read a field from dicts, typed SDK models, or model extras.

        Provides a unified way to access a named field regardless of whether
        the object is a plain dictionary, a Pydantic model (as used by the
        OpenAI SDK), or an object with ``model_extra`` for additional fields
        not declared in the schema.

        The lookup order is:
        1. ``dict.get(field)`` if ``obj`` is a dict
        2. ``getattr(obj, field)`` for regular attributes
        3. ``obj.model_extra.get(field)`` for Pydantic model extras

        Args:
            obj: The object to read from. Can be ``None``, a dict, a Pydantic
                model, or any object with attributes.
            field: The name of the field to retrieve.

        Returns:
            The field value if found, or ``None`` if the object is ``None``
            or the field does not exist in any of the checked locations."""
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
        """Convert provider-specific reasoning payloads into plain text.

        Recursively processes reasoning content from various formats returned
        by different OpenAI-compatible providers. Reasoning models (o1, o3,
        DeepSeek-R1, etc.) may embed chain-of-thought tokens in different
        structures: strings, lists, dicts, or Pydantic SDK objects.

        The method attempts to extract text by:
        1. Returning strings directly
        2. Recursively joining list/tuple elements
        3. Searching dicts for known text keys (``text``, ``content``,
           ``summary_text``, ``reasoning_content``, etc.)
        4. Falling back to attribute access via :meth:`_get_openai_field`
           for SDK model objects

        Args:
            value: The reasoning payload to convert. Can be ``None``, a string,
                a list/tuple of items, a dict with known text keys, or an
                SDK model object with reasoning attributes.

        Returns:
            The extracted plain text reasoning content. Returns an empty
            string if ``value`` is ``None`` or no text can be extracted."""
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
        """Extract reasoning content from a chat-completions style message.

        Searches the message object for reasoning tokens that reasoning models
        (o1, o3, DeepSeek-R1, etc.) include alongside the main content. The
        method checks multiple possible field names used by different providers.

        The extraction strategy is:
        1. Check top-level fields: ``reasoning_content``, ``reasoning``,
           ``delta_reasoning`` on the message object
        2. If the message ``content`` is a list, scan items for those whose
           ``type`` contains ``'reasoning'`` and extract their text

        Args:
            message: A chat completion message object (from
                ``response.choices[0].message``). Can be a Pydantic model or
                dict with reasoning-related fields.

        Returns:
            The extracted reasoning text as a single string. Returns an empty
            string if no reasoning content is found in the message."""
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
        """Extract reasoning deltas from streaming chat chunks and Responses API events.

        Processes a single streaming chunk to extract any incremental reasoning
        content. Supports multiple event formats used by the OpenAI Chat
        Completions API and the Responses API for reasoning models.

        The extraction checks (in order):
        1. Responses API event types: ``response.reasoning.delta``,
           ``response.reasoning_text.delta``, etc.
        2. Chat completions delta fields: ``reasoning_content``, ``reasoning``,
           ``delta_reasoning`` on ``choices[0].delta``
        3. Top-level ``delta_reasoning`` field on the chunk itself

        Args:
            chunk: A streaming chunk object from either the Chat Completions
                or Responses API. Can be a Pydantic model, dict, or any
                object with the expected attributes.

        Returns:
            The reasoning text delta from this chunk. Returns an empty string
            if the chunk contains no reasoning content."""
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
        """Extract text content from an OpenAI ChatCompletion response.

        Safely extracts the message content from the first choice in an OpenAI
        response. Handles both regular text responses and tool call responses
        (which have no text content).

        The method checks for:
        1. Presence of choices array
        2. Message object in first choice
        3. Content attribute on message
        4. Tool calls (returns empty string if present without content)

        Args:
            response: OpenAI ChatCompletion response object. Should have a
                'choices' attribute containing at least one choice with a
                'message' attribute.

        Returns:
            The text content from the response message. Returns an empty string
            if the response is a tool call response, has no content, or if the
            response structure is unexpected.

        Example:
            response = await llm.generate_completion("Hello!")
            content = llm.extract_content(response)
            print(content)  # "Hello! How can I help you today?"

            # Tool call response returns empty string
            response = await llm.generate_completion(prompt, tools=tools)
            content = llm.extract_content(response)  # "" (tool calls present)
        """
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message

            if hasattr(message, "content") and message.content:
                return message.content

            if hasattr(message, "tool_calls") and message.tool_calls:
                return ""

        return ""

    def extract_reasoning_content(self, response: Any) -> str:
        """Extract reasoning/thinking content from an OpenAI response.

        Extracts reasoning tokens from reasoning models (o1, o3, etc.)
        in non-streaming responses. These models may include internal
        chain-of-thought reasoning separately from the main content.

        Args:
            response: OpenAI ChatCompletion response object.

        Returns:
            The reasoning content string, or empty string if not present.

        Example:
            response = await llm.generate_completion("Solve this math problem")
            reasoning = llm.extract_reasoning_content(response)
            if reasoning:
                print(f"Model reasoning: {reasoning}")
            content = llm.extract_content(response)
            print(f"Answer: {content}")
        """
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            reasoning = self._extract_reasoning_from_message(message)
            if reasoning:
                return reasoning
        return ""

    async def process_streaming_response(self, response: Any, callback: Callable[[str, Any], None]) -> str:
        """Process a streaming response from OpenAI with a callback for each chunk.

        Iterates through all chunks in a streaming response, extracts text content
        and reasoning tokens from each delta, accumulates the full response, and
        invokes the callback for each content chunk received.

        This method is useful for displaying streaming output in real-time while
        also capturing the complete response. The callback receives both the
        individual chunk content and the raw chunk object for additional processing.

        Note: Reasoning tokens (from o1/o3 models) are included in the callback
        via the raw chunk object. Access them via chunk.choices[0].delta.reasoning_content.

        Args:
            response: OpenAI streaming response iterator. Should be the result of
                a chat.completions.create() call with stream=True.
            callback: Function called for each chunk with (content, raw_chunk).
                - content: The text content from this chunk's delta
                - raw_chunk: The raw ChatCompletionChunk object

        Returns:
            The complete accumulated content from all chunks concatenated together.

        Example:
            def print_chunk(content: str, chunk: Any) -> None:
                print(content, end="", flush=True)

            stream = await llm.generate_completion("Tell me a story", stream=True)
            full_text = await llm.process_streaming_response(stream, print_chunk)
            print()  # Newline after streaming
            print(f"Total length: {len(full_text)}")
        """
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
        """Stream completion chunks with tool/function call detection and accumulation.

        Iterates through an OpenAI streaming response, yielding structured data
        for each chunk. Automatically detects and accumulates tool calls across
        chunks, building complete function call information as it streams.

        This method handles the complexity of OpenAI's streaming format where
        tool call arguments are split across multiple chunks. It maintains
        internal accumulators and provides both incremental updates and
        accumulated totals in each yielded chunk.

        Args:
            response: OpenAI streaming response iterator from a chat.completions
                call with stream=True.
            agent: Optional agent instance for function detection. Currently
                reserved for future use with agent-specific function filtering.

        Yields:
            Dictionary containing streaming chunk information with keys:
                - content: Text content in this specific chunk (or None)
                - buffered_content: All accumulated text content so far
                - function_calls: List of completed function calls (populated
                  only in final chunk when tool calls are present)
                - tool_calls: Accumulated tool call data indexed by position
                - streaming_tool_calls: Incremental tool call updates in this chunk
                - raw_chunk: The original OpenAI ChatCompletionChunk object
                - is_final: Boolean indicating if this is the last chunk

        Example:
            stream = await llm.generate_completion(prompt, stream=True, tools=tools)
            for chunk in llm.stream_completion(stream):
                # Print text content as it arrives
                if chunk["content"]:
                    print(chunk["content"], end="")

                # Check for completed function calls at end
                if chunk["is_final"] and chunk["function_calls"]:
                    for call in chunk["function_calls"]:
                        print(f"Function: {call['name']}")
                        print(f"Args: {call['arguments']}")
        """
        buffered_content = ""
        buffered_reasoning_content = ""
        function_calls = []
        tool_call_accumulator = {}

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

    async def astream_completion(
        self,
        response: Any,
        agent: Any | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async stream completion chunks with tool/function call detection.

        Asynchronous version of stream_completion that works with async iterators.
        Iterates through an OpenAI async streaming response, yielding structured
        data for each chunk with automatic tool call accumulation.

        This method is designed for use with AsyncOpenAI client or when the
        streaming response is an async iterator. It provides the same functionality
        as stream_completion but with proper async/await semantics.

        Args:
            response: OpenAI async streaming response (AsyncIterator). Should be
                the result of an async chat.completions.create() call with stream=True.
            agent: Optional agent instance for function detection. Currently
                reserved for future use with agent-specific function filtering.

        Yields:
            Dictionary containing streaming chunk information with keys:
                - content: Text content in this specific chunk (or None)
                - buffered_content: All accumulated text content so far
                - function_calls: List of completed function calls (populated
                  only in final chunk when tool calls are present)
                - tool_calls: Accumulated tool call data indexed by position
                - streaming_tool_calls: Incremental tool call updates in this chunk
                - raw_chunk: The original OpenAI ChatCompletionChunk object
                - is_final: Boolean indicating if this is the last chunk

        Example:
            stream = await async_llm.generate_completion(prompt, stream=True)
            async for chunk in llm.astream_completion(stream):
                if chunk["content"]:
                    print(chunk["content"], end="")

                if chunk["is_final"]:
                    print()  # Newline at end
        """
        buffered_content = ""
        buffered_reasoning_content = ""
        function_calls = []
        tool_call_accumulator = {}

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

    def parse_tool_calls(self, raw_data: Any) -> list[dict[str, Any]]:
        """Parse tool/function calls from an OpenAI response message.

        Extracts and standardizes tool call information from an OpenAI message
        object. Converts OpenAI's tool call format to a simplified dictionary
        format used consistently across all Calute LLM providers.

        This method is typically used to extract tool calls from non-streaming
        responses where the complete tool call data is available at once.

        Args:
            raw_data: OpenAI message object (typically response.choices[0].message)
                that may contain a 'tool_calls' attribute with a list of tool
                call objects.

        Returns:
            List of standardized tool call dictionaries, each containing:
                - id: Unique identifier for the tool call
                - name: Name of the function to call
                - arguments: JSON string of function arguments
            Returns an empty list if no tool calls are present.

        Example:
            response = await llm.generate_completion(prompt, tools=tools)
            message = response.choices[0].message
            tool_calls = llm.parse_tool_calls(message)
            for call in tool_calls:
                print(f"Call {call['id']}: {call['name']}")
                args = json.loads(call['arguments'])
                # Execute the function with args...
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

    def fetch_model_info(self) -> dict[str, Any]:
        """Fetch model metadata from the OpenAI /v1/models endpoint.

        Queries the OpenAI models API to retrieve information about the configured
        model. This information can include context window size, capabilities,
        and other metadata depending on the API endpoint.

        The method searches through all available models returned by the API
        to find the one matching self.config.model, then extracts relevant
        metadata fields.

        Returns:
            Dictionary containing model metadata:
                - max_model_len: Maximum context length in tokens (if available)
                - metadata: Additional model metadata dictionary
            Returns an empty dictionary if the model is not found or if the
            API call fails.

        Note:
            This method silently catches all exceptions to prevent initialization
            failures. Some OpenAI-compatible endpoints may not support the
            /v1/models endpoint or may return different metadata fields.

        Example:
            llm = OpenAILLM(model="gpt-4")
            info = llm.fetch_model_info()
            if info.get("max_model_len"):
                print(f"Context window: {info['max_model_len']} tokens")
        """
        try:
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
        """Close the OpenAI client and release resources.

        Closes any open HTTP connections maintained by the OpenAI client.
        This method is called automatically when using the LLM provider as
        an async context manager.

        The method safely checks if the client has a close method before
        calling it, ensuring compatibility with different client versions
        or custom client implementations.

        Side Effects:
            - Closes the underlying HTTP client/session
            - Releases any connection pool resources

        Example:
            llm = OpenAILLM(model="gpt-4")
            try:
                response = await llm.generate_completion("Hello")
            finally:
                await llm.close()

            # Or use as context manager (preferred):
            async with OpenAILLM(model="gpt-4") as llm:
                response = await llm.generate_completion("Hello")
        """
        if hasattr(self.async_client, "aclose"):
            await self.async_client.aclose()
        elif hasattr(self.async_client, "close"):
            maybe_result = self.async_client.close()
            if hasattr(maybe_result, "__await__"):
                await maybe_result

        if hasattr(self.client, "close"):
            maybe_result = self.client.close()
            if hasattr(maybe_result, "__await__"):
                await maybe_result
