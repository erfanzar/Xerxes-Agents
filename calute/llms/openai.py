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

import os
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from .base import BaseLLM, LLMConfig


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

    def __init__(self, config: LLMConfig | None = None, client: Any | None = None, **kwargs):
        """Initialize the OpenAI LLM provider.

        Creates a new OpenAI LLM provider instance with the specified configuration.
        Supports three initialization patterns: explicit config, keyword arguments,
        or injected client instance.

        Args:
            config: LLM configuration object. If None, a default config is created
                using the provided kwargs with model defaulting to "gpt-4o-mini".
            client: Optional pre-configured OpenAI client instance. When provided,
                the client is used directly without creating a new one. Useful for
                custom authentication or connection pooling scenarios.
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

    async def process_streaming_response(self, response: Any, callback: Callable[[str, Any], None]) -> str:
        """Process a streaming response from OpenAI with a callback for each chunk.

        Iterates through all chunks in a streaming response, extracts text content
        from each delta, accumulates the full response, and invokes the callback
        for each content chunk received.

        This method is useful for displaying streaming output in real-time while
        also capturing the complete response. The callback receives both the
        individual chunk content and the raw chunk object for additional processing.

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
        if hasattr(self.client, "close"):
            self.client.close()
