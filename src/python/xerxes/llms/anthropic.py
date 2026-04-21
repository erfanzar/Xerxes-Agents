# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""Anthropic Claude LLM provider implementation.

This module provides integration with Anthropic's Claude API for the Xerxes
framework. It implements the BaseLLM interface to offer synchronous and
asynchronous completion generation, streaming responses, and function/tool
call support for Claude models.

The implementation uses httpx for HTTP communication with the Anthropic API,
providing efficient async operations and streaming support. It handles
message format conversion from OpenAI-style to Anthropic's expected format.

Features:
- Support for all Claude 3.x and 4.x models (Opus, Sonnet, Haiku)
- Streaming responses with Server-Sent Events (SSE) parsing
- Tool/function call support with structured output parsing
- Automatic message format conversion from OpenAI-style to Anthropic format
- Context length metadata for supported models
- Async context manager support for proper resource cleanup

Supported models include:
- claude-3-opus-20240229 (200K context)
- claude-3-sonnet-20240229 (200K context)
- claude-3-haiku-20240307 (200K context)
- claude-3-5-sonnet-20240620 (200K context)
- claude-3-5-haiku-20241022 (200K context)
- claude-opus-4-20250514 (200K context)
- claude-sonnet-4-20250514 (200K context)

Typical usage example:
    from xerxes.llms.anthropic import AnthropicLLM
    from xerxes.llms.base import LLMConfig

    config = LLMConfig(
        model="claude-3-opus-20240229",
        temperature=0.7,
        max_tokens=4096,
        api_key="your-api-key"
    )

    async with AnthropicLLM(config) as llm:
        response = await llm.generate_completion("Hello, Claude!")
        content = llm.extract_content(response)
        print(content)

Note:
    Requires the httpx library for HTTP communication.
    Install with: pip install httpx
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from .base import BaseLLM, LLMConfig

ANTHROPIC_CONTEXT_LENGTHS = {
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-3-5-sonnet": 200000,
    "claude-3-5-haiku": 200000,
    "claude-opus-4": 200000,
    "claude-sonnet-4": 200000,
}

httpx: Any
try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    httpx = None


class AnthropicLLM(BaseLLM):
    """Anthropic Claude LLM provider implementation.

    AnthropicLLM provides integration with Anthropic's Claude API, implementing
    the BaseLLM interface for seamless integration with the Xerxes framework.
    It supports all Claude 3.x and 4.x model variants with features including
    streaming responses, tool/function calling, and automatic message format
    conversion.

    This implementation uses httpx for async HTTP communication and handles
    the conversion between OpenAI-style message formats and Anthropic's
    expected format. System messages are automatically merged into the first
    user message as Anthropic requires.

    Attributes:
        config: LLMConfig instance containing provider configuration.
        version: Anthropic API version string (e.g., "2023-06-01").
        client: httpx.AsyncClient instance for making API requests.

    Example:

        config = LLMConfig(
            model="claude-3-opus-20240229",
            temperature=0.5,
            max_tokens=4096,
            api_key="sk-ant-..."
        )
        async with AnthropicLLM(config) as llm:
            response = await llm.generate_completion("Hello!")
            print(llm.extract_content(response))


        llm = AnthropicLLM(model="claude-3-haiku-20240307", api_key="sk-ant-...")
        response = await llm.generate_completion("What is 2+2?")
        await llm.close()

    Note:
        The API key can be provided via config, kwargs, or the
        ANTHROPIC_API_KEY environment variable.
    """

    def __init__(self, config: LLMConfig | None = None, version: str = "2023-06-01", **kwargs):
        """Initialize the Anthropic Claude LLM provider.

        Creates a new AnthropicLLM instance configured for the Claude API.
        If no config is provided, a default configuration is created using
        the provided keyword arguments with sensible defaults.

        Args:
            config: LLM configuration object. If None, a default config is
                created using kwargs. Defaults to claude-3-opus-20240229 model.
            version: Anthropic API version header value. Different versions
                may have different features and behavior. Defaults to
                "2023-06-01" which is a stable API version.
            **kwargs: Additional configuration parameters when config is None:
                - model: Model identifier (default: "claude-3-opus-20240229")
                - api_key: API key for authentication
                - base_url: Base URL for API (default: "https://api.anthropic.com")
                - temperature, max_tokens, top_p, etc.

        Raises:
            ImportError: If the httpx library is not installed.

        Side Effects:
            - Stores configuration in self.config
            - Sets self.version to the API version
            - Calls _initialize_client() to set up the HTTP client
        """
        if not HAS_HTTPX:
            raise ImportError("httpx library required for Anthropic. Install with: pip install httpx")

        if config is None:
            config = LLMConfig(
                model=kwargs.pop("model", "claude-3-opus-20240229"),
                api_key=kwargs.pop("api_key", None),
                base_url=kwargs.pop("base_url", "https://api.anthropic.com"),
                **kwargs,
            )

        self.version = version
        self.client: httpx.AsyncClient | None = None
        super().__init__(config)

    def _initialize_client(self) -> None:
        """Initialize the Anthropic HTTP client.

        Sets up the httpx.AsyncClient with appropriate headers for Anthropic
        API authentication and versioning. The client is configured with the
        base URL from config and appropriate timeouts.

        The API key is retrieved from the config, or falls back to the
        ANTHROPIC_API_KEY environment variable if not explicitly provided.

        Raises:
            ValueError: If no API key is provided via config or environment.

        Side Effects:
            - Creates and stores httpx.AsyncClient in self.client
            - Calls _auto_fetch_model_info() to populate model metadata
        """
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided")

        self.client = httpx.AsyncClient(
            base_url=self.config.base_url or "",
            headers={
                "anthropic-version": self.version,
                "x-api-key": api_key,
                "content-type": "application/json",
            },
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
        """Generate a completion using the Anthropic Claude API.

        Sends a prompt to the Claude API and returns the model's response.
        Supports both simple string prompts and OpenAI-style message lists.
        When using message lists, the method automatically converts them to
        Anthropic's expected format, handling system messages appropriately.

        Args:
            prompt: The input prompt. Can be either:
                - A string: Converted to a single user message
                - A list of message dicts with 'role' and 'content' keys,
                  following OpenAI's chat format. Roles can be 'user',
                  'assistant', or 'system'.
            model: Model identifier to use, overriding config.model.
                Example: "claude-3-opus-20240229"
            temperature: Sampling temperature (0.0-1.0), overriding config.
                Higher values produce more random outputs.
            max_tokens: Maximum tokens to generate, overriding config.
                Claude requires this parameter (unlike some other providers).
            top_p: Nucleus sampling parameter, overriding config.
                Only applied if different from the default 0.95.
            stop: List of stop sequences that will halt generation,
                overriding config.stop. Called "stop_sequences" in Anthropic API.
            stream: Whether to stream the response, overriding config.stream.
                If True, returns an async iterator of chunks.
            **kwargs: Additional Anthropic-specific parameters passed directly
                to the API, such as:
                - tools: List of tool definitions for function calling
                - tool_choice: How to select tools
                - metadata: Request metadata

        Returns:
            If stream is False: A dictionary containing the API response with:
                - id: Unique message ID
                - type: "message"
                - role: "assistant"
                - content: List of content blocks (text, tool_use)
                - model: Model used
                - stop_reason: Why generation stopped
                - usage: Token usage statistics

            If stream is True: An AsyncIterator yielding SSE event dicts.

        Raises:
            RuntimeError: If the API request fails due to network errors,
                authentication issues, or API errors.

        Example:

            response = await llm.generate_completion("What is Python?")


            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]
            response = await llm.generate_completion(messages)


            async for chunk in await llm.generate_completion("Hi", stream=True):
                print(chunk)
        """

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = self._convert_messages(prompt)

        payload = {
            "model": model or self.config.model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }

        if top_p is not None or self.config.top_p != 0.95:
            payload["top_p"] = top_p if top_p is not None else self.config.top_p

        if stop or self.config.stop:
            payload["stop_sequences"] = stop or self.config.stop

        payload.update(kwargs)
        payload.update(self.config.extra_params)

        use_stream = stream if stream is not None else self.config.stream

        try:
            if use_stream:
                return self._stream_completion(payload)
            else:
                assert self.client is not None
                response = await self.client.post("/v1/messages", json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Anthropic API request failed: {e}") from e

    async def _stream_completion(self, payload: dict) -> AsyncIterator[dict]:
        """Stream completion chunks from the Anthropic API.

        Internal method that handles streaming HTTP responses from the
        Anthropic Messages API. Uses Server-Sent Events (SSE) format to
        parse incremental response chunks.

        The method streams the response line by line, parsing SSE data
        events and yielding parsed JSON chunks. The "[DONE]" sentinel
        is filtered out from the output.

        Args:
            payload: Request payload dictionary to send to the API.
                The "stream" key is automatically set to True.

        Yields:
            dict: Parsed JSON event data from each SSE data line.
                Event types include:
                - message_start: Initial message metadata
                - content_block_start: Start of a content block
                - content_block_delta: Incremental content update
                - content_block_stop: End of a content block
                - message_delta: Message-level updates (stop_reason, usage)
                - message_stop: End of the message stream

        Raises:
            httpx.HTTPError: If the HTTP request fails or returns an error.

        Note:
            This is an internal method called by generate_completion()
            when streaming is enabled. Use generate_completion(stream=True)
            instead of calling this directly.
        """
        payload["stream"] = True
        assert self.client is not None

        async with self.client.stream("POST", "/v1/messages", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data != "[DONE]":
                        yield json.loads(data)

    def _convert_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Convert OpenAI-style messages to Anthropic format.

        Transforms a list of messages from OpenAI's chat format to Anthropic's
        expected format. The key difference is that Anthropic doesn't support
        a separate "system" role - system messages must be prepended to the
        first user message.

        The conversion handles:
        - Extracting system messages and merging them with the first user message
        - Filtering out messages with unsupported roles
        - Preserving the order of user and assistant messages

        Args:
            messages: List of message dictionaries in OpenAI format, each with:
                - role: "user", "assistant", or "system"
                - content: The message text

        Returns:
            List of message dictionaries in Anthropic format, containing only
            "user" and "assistant" roles. System content is prepended to the
            first user message with a double newline separator.

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are helpful."},
            ...     {"role": "user", "content": "Hello!"}
            ... ]
            >>> converted = llm._convert_messages(messages)
            >>>

        Note:
            If there's a system message but no user message, or if the first
            non-system message is from the assistant, the system content is
            inserted as a user message at the beginning.
        """
        converted = []
        system_content = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_content = content
            else:
                if role in ["user", "assistant"]:
                    converted.append({"role": role, "content": content})

        if system_content and converted:
            if converted[0]["role"] == "user":
                converted[0]["content"] = f"{system_content}\n\n{converted[0]['content']}"
            else:
                converted.insert(0, {"role": "user", "content": system_content})

        return converted

    def extract_content(self, response: Any) -> str:
        """Extract text content from an Anthropic API response.

        Parses the response from the Anthropic Messages API and extracts
        all text content from the content blocks. Anthropic responses contain
        a list of content blocks, each with a type (usually "text" or "tool_use").
        This method concatenates all text blocks into a single string.

        Args:
            response: The API response dictionary from generate_completion().
                Expected structure:
                {
                    "content": [
                        {"type": "text", "text": "Hello!"},
                        {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
                    ],
                    ...
                }

        Returns:
            Concatenated text from all "text" type content blocks.
            Returns an empty string if:
            - response is not a dictionary
            - response has no "content" key
            - no text blocks are present

        Example:
            >>> response = {
            ...     "content": [
            ...         {"type": "text", "text": "Hello, "},
            ...         {"type": "text", "text": "world!"}
            ...     ]
            ... }
            >>> llm.extract_content(response)
            "Hello, world!"
        """
        if isinstance(response, dict):
            content = response.get("content", [])
            if content and isinstance(content, list):
                text_parts = []
                for block in content:
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                return "".join(text_parts)
        return ""

    async def process_streaming_response(
        self,
        response: Any,
        callback: Callable[[str, Any], None],
    ) -> str:
        """Process a streaming response from the Anthropic API.

        Iterates through streaming response events and accumulates text content.
        For each content delta event, the callback is invoked with the new text
        and the raw event data.

        This method is useful for real-time display of streaming output while
        also collecting the complete response.

        Args:
            response: An async iterator of streaming events from
                generate_completion(stream=True). Events are dictionaries
                with a "type" field indicating the event type.
            callback: A function called for each text chunk received.
                Signature: callback(text: str, raw_chunk: dict)
                - text: The incremental text content
                - raw_chunk: The raw event dictionary

        Returns:
            The complete accumulated text content from all content_block_delta
            events in the stream.

        Example:
            def on_chunk(text, chunk):
                print(text, end="", flush=True)

            response = await llm.generate_completion("Tell me a story", stream=True)
            full_text = await llm.process_streaming_response(response, on_chunk)
            print(f"\\nTotal: {len(full_text)} chars")
        """
        accumulated_content = ""

        async for chunk in response:
            if chunk.get("type") == "content_block_delta":
                delta = chunk.get("delta", {})
                if text := delta.get("text"):
                    accumulated_content += text
                    callback(text, chunk)

        return accumulated_content

    def stream_completion(
        self,
        response: Any,
        agent: Any | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream completion chunks with function call detection.

        Processes a synchronous streaming response from Anthropic, yielding
        standardized chunk dictionaries that include accumulated content and
        detected function/tool calls. This method provides a unified interface
        for handling streaming responses across different providers.

        The method tracks:
        - Incremental text content from content_block_delta events
        - Accumulated content across the entire stream
        - Tool use blocks for function calling
        - Stream completion (message_stop event)

        Args:
            response: A synchronous iterator of streaming events from
                the Anthropic API. Each event is either a dictionary or
                an object with a "type" attribute.
            agent: Optional agent instance for advanced function detection.
                Currently reserved for future use.

        Yields:
            dict: Standardized chunk information with keys:
                - content: Text content in this chunk (str or None)
                - buffered_content: All text accumulated so far (str)
                - function_calls: List of detected function calls (list)
                - tool_calls: Raw tool call data (None for Anthropic)
                - raw_chunk: The original event data
                - is_final: True if this is the final chunk (bool)

            Function call format in function_calls list:
                {
                    "id": "tool_use_id",
                    "name": "function_name",
                    "arguments": '{"arg": "value"}'
                }

        Example:
            for chunk in llm.stream_completion(response):
                if chunk["content"]:
                    print(chunk["content"], end="")
                if chunk["is_final"]:
                    for call in chunk["function_calls"]:
                        print(f"Function: {call['name']}")
        """
        buffered_content = ""
        function_calls: list[dict[str, Any]] = []

        for event in response:
            chunk_data = {
                "content": None,
                "buffered_content": buffered_content,
                "function_calls": [],
                "tool_calls": None,
                "raw_chunk": event,
                "is_final": False,
            }

            event_type = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)

            if event_type == "content_block_delta":
                delta = event.get("delta", {}) if isinstance(event, dict) else getattr(event, "delta", {})
                text = delta.get("text", "") if isinstance(delta, dict) else getattr(delta, "text", "")
                if text:
                    buffered_content += text
                    chunk_data["content"] = text
                    chunk_data["buffered_content"] = buffered_content
            elif event_type == "message_stop":
                chunk_data["is_final"] = True
                chunk_data["function_calls"] = function_calls
            elif event_type == "tool_use":
                name = event.get("name") if isinstance(event, dict) else getattr(event, "name", None)
                input_data = event.get("input") if isinstance(event, dict) else getattr(event, "input", None)
                if name:
                    function_calls.append(
                        {
                            "id": event.get("id") if isinstance(event, dict) else getattr(event, "id", None),
                            "name": name,
                            "arguments": json.dumps(input_data) if input_data else "",
                        }
                    )

            yield chunk_data

    def astream_completion(
        self,
        response: Any,
        agent: Any | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async stream completion chunks with function call detection.

        Asynchronous version of stream_completion() that processes streaming
        responses from the Anthropic API. Yields standardized chunk dictionaries
        containing accumulated content and detected function/tool calls.

        This method is the preferred way to handle streaming responses in
        async contexts, providing the same unified interface as the synchronous
        stream_completion() method.

        The method tracks:
        - Incremental text content from content_block_delta events
        - Accumulated content across the entire stream
        - Tool use blocks for function calling
        - Stream completion (message_stop event)

        Args:
            response: An async iterator of streaming events from
                generate_completion(stream=True). Each event is either a
                dictionary or an object with a "type" attribute.
            agent: Optional agent instance for advanced function detection.
                Currently reserved for future use.

        Yields:
            dict: Standardized chunk information with keys:
                - content: Text content in this chunk (str or None)
                - buffered_content: All text accumulated so far (str)
                - function_calls: List of detected function calls (list)
                - tool_calls: Raw tool call data (None for Anthropic)
                - raw_chunk: The original event data
                - is_final: True if this is the final chunk (bool)

            Function call format in function_calls list:
                {
                    "id": "tool_use_id",
                    "name": "function_name",
                    "arguments": '{"arg": "value"}'
                }

        Example:
            response = await llm.generate_completion("Hello", stream=True)
            async for chunk in llm.astream_completion(response):
                if chunk["content"]:
                    print(chunk["content"], end="", flush=True)
                if chunk["is_final"] and chunk["function_calls"]:
                    print("\\nFunction calls detected!")
        """

        async def _gen() -> AsyncIterator[dict[str, Any]]:
            buffered_content = ""
            function_calls: list[dict[str, Any]] = []

            async for event in response:
                chunk_data = {
                    "content": None,
                    "buffered_content": buffered_content,
                    "function_calls": [],
                    "tool_calls": None,
                    "raw_chunk": event,
                    "is_final": False,
                }

                event_type = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)

                if event_type == "content_block_delta":
                    delta = event.get("delta", {}) if isinstance(event, dict) else getattr(event, "delta", {})
                    text = delta.get("text", "") if isinstance(delta, dict) else getattr(delta, "text", "")
                    if text:
                        buffered_content += text
                        chunk_data["content"] = text
                        chunk_data["buffered_content"] = buffered_content
                elif event_type == "message_stop":
                    chunk_data["is_final"] = True
                    chunk_data["function_calls"] = function_calls
                elif event_type == "tool_use":
                    name = event.get("name") if isinstance(event, dict) else getattr(event, "name", None)
                    input_data = event.get("input") if isinstance(event, dict) else getattr(event, "input", None)
                    if name:
                        function_calls.append(
                            {
                                "id": event.get("id") if isinstance(event, dict) else getattr(event, "id", None),
                                "name": name,
                                "arguments": json.dumps(input_data) if input_data else "",
                            }
                        )

                yield chunk_data

        return _gen()

    def parse_tool_calls(self, raw_data: Any) -> list[dict[str, Any]]:
        """Parse tool/function calls from an Anthropic API response.

        Extracts tool use blocks from an Anthropic response and converts them
        to a standardized format compatible with the Xerxes framework. This
        method handles Anthropic's content block structure where tool calls
        are embedded as "tool_use" type blocks.

        Args:
            raw_data: The API response dictionary from generate_completion().
                Expected to have a "content" key containing a list of blocks:
                {
                    "content": [
                        {"type": "text", "text": "..."},
                        {
                            "type": "tool_use",
                            "id": "toolu_123",
                            "name": "get_weather",
                            "input": {"location": "NYC"}
                        }
                    ]
                }

        Returns:
            List of standardized tool call dictionaries, each containing:
                - id: The tool use ID from Anthropic
                - name: The function/tool name
                - arguments: JSON string of the input parameters

            Returns an empty list if:
            - raw_data is not a dictionary
            - raw_data has no "content" key
            - no tool_use blocks are present

        Example:
            >>> response = await llm.generate_completion(prompt, tools=[...])
            >>> tool_calls = llm.parse_tool_calls(response)
            >>> for call in tool_calls:
            ...     args = json.loads(call["arguments"])
            ...     result = execute_function(call["name"], args)
        """
        tool_calls = []
        if isinstance(raw_data, dict) and "content" in raw_data:
            for block in raw_data["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id"),
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        }
                    )
        return tool_calls

    def fetch_model_info(self) -> dict[str, Any]:
        """Fetch model metadata from known Anthropic model specifications.

        Since Anthropic does not provide a public API endpoint for querying
        model capabilities, this method uses a local mapping of known model
        context lengths (ANTHROPIC_CONTEXT_LENGTHS). The model is matched
        by prefix to handle versioned model names.

        The context length information is useful for:
        - Token counting and context window management
        - Preventing context overflow errors
        - Optimizing prompt construction

        Returns:
            Dictionary with model metadata. Contains:
                - max_model_len: Maximum context length in tokens (int)

            Returns an empty dictionary if the model name doesn't match
            any known prefixes.

        Example:
            >>> llm = AnthropicLLM(model="claude-3-opus-20240229")
            >>> info = llm.fetch_model_info()
            >>> print(info)

        Note:
            This method is called automatically during client initialization
            via _auto_fetch_model_info(). The result is stored in
            config.model_metadata and config.max_model_len.
        """
        model = self.config.model
        for prefix, context_len in ANTHROPIC_CONTEXT_LENGTHS.items():
            if model.startswith(prefix):
                return {"max_model_len": context_len}
        return {}

    async def close(self) -> None:
        """Close the HTTP client and release resources.

        Properly closes the httpx.AsyncClient connection pool. This method
        should be called when done using the LLM provider to prevent resource
        leaks. It is called automatically when using the provider as an async
        context manager.

        This method is safe to call multiple times - it checks for client
        existence before attempting to close.

        Example:
            llm = AnthropicLLM(config)
            try:
                response = await llm.generate_completion("Hello")
            finally:
                await llm.close()


            async with AnthropicLLM(config) as llm:
                response = await llm.generate_completion("Hello")
        """
        if self.client:
            await self.client.aclose()
