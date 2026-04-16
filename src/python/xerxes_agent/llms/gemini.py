# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
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


"""Google Gemini LLM provider implementation.

This module provides integration with Google's Generative AI (Gemini) models
through the google-generativeai Python SDK. It supports both streaming and
non-streaming completions, multi-turn conversations, and function calling.

The module handles:
- Authentication via API key (config, environment variables GEMINI_API_KEY
  or GOOGLE_API_KEY)
- Message format conversion from standard chat format to Gemini format
- Streaming response processing with callback support
- Automatic model metadata fetching (token limits)
- Tool/function call parsing from Gemini responses

Supported models include:
- gemini-pro (default)
- gemini-pro-vision
- gemini-1.5-pro
- gemini-1.5-flash
- Other models available through the Gemini API

Typical usage example:
    from xerxes_agent.llms.gemini import GeminiLLM
    from xerxes_agent.llms.base import LLMConfig

    config = LLMConfig(
        model="gemini-1.5-pro",
        temperature=0.7,
        max_tokens=2048,
        api_key="your-api-key"
    )

    async with GeminiLLM(config) as llm:
        response = await llm.generate_completion("Explain quantum computing")
        content = llm.extract_content(response)
        print(content)

Note:
    Requires the google-generativeai package to be installed:
    pip install google-generativeai
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from .base import BaseLLM, LLMConfig


class GeminiLLM(BaseLLM):
    """Google Gemini LLM provider implementation.

    GeminiLLM provides a complete integration with Google's Generative AI
    (Gemini) API, offering text generation capabilities with support for
    both single prompts and multi-turn conversations.

    This implementation handles the conversion between the standardized
    Xerxes message format and Gemini's expected input format, manages
    streaming responses, and supports function calling for agentic workflows.

    Attributes:
        config: LLMConfig instance containing provider configuration.
        client: Google GenerativeModel client instance for API calls.
        genai: Reference to the google.generativeai module for configuration
            and model access.

    Example:
        # Basic usage with string prompt
        llm = GeminiLLM(model="gemini-pro", api_key="your-key")
        response = await llm.generate_completion("What is the meaning of life?")
        print(llm.extract_content(response))

        # Using with chat-style messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        response = await llm.generate_completion(messages)

        # Streaming response
        response = await llm.generate_completion("Tell me a story", stream=True)
        for chunk in llm.stream_completion(response):
            print(chunk["content"], end="", flush=True)

    Note:
        The Gemini API key can be provided via:
        1. The config.api_key parameter
        2. The GEMINI_API_KEY environment variable
        3. The GOOGLE_API_KEY environment variable
    """

    def __init__(self, config: LLMConfig | None = None, client: Any | None = None, **kwargs):
        """Initialize the Gemini LLM provider.

        Creates a new GeminiLLM instance with the specified configuration.
        If no config is provided, a default configuration is created using
        the keyword arguments with sensible defaults for Gemini models.

        Args:
            config: LLM configuration object. If None, a default config is
                created with model="gemini-pro" and any provided kwargs.
            client: Optional pre-initialized GenerativeModel client instance.
                If provided, this client will be used instead of creating
                a new one during initialization.
            **kwargs: Additional configuration parameters passed to LLMConfig
                when config is None. Common kwargs include:
                - model: Model name (default: "gemini-pro")
                - api_key: Gemini API key
                - temperature: Sampling temperature
                - max_tokens: Maximum output tokens

        Raises:
            ImportError: If google-generativeai package is not installed.
            ValueError: If no API key is provided or found in environment.

        Example:
            # Using explicit config
            config = LLMConfig(model="gemini-1.5-pro", api_key="key")
            llm = GeminiLLM(config)

            # Using kwargs for convenience
            llm = GeminiLLM(model="gemini-pro", api_key="key", temperature=0.5)

            # Using pre-initialized client
            import google.generativeai as genai
            client = genai.GenerativeModel("gemini-pro")
            llm = GeminiLLM(client=client)
        """

        if config is None:
            config = LLMConfig(model=kwargs.pop("model", "gemini-pro"), api_key=kwargs.pop("api_key", None), **kwargs)

        self.client = client
        self.genai = None
        super().__init__(config)

    def _initialize_client(self) -> None:
        """Initialize the Gemini client and configure API access.

        Sets up the Google GenerativeAI module with the provided API key
        and creates a GenerativeModel client for the configured model.
        This method is called automatically at the end of __init__.

        The API key is resolved in the following order:
        1. config.api_key (explicitly provided)
        2. GEMINI_API_KEY environment variable
        3. GOOGLE_API_KEY environment variable

        Side Effects:
            - Imports and stores google.generativeai module in self.genai
            - Configures the genai module with the API key
            - Creates GenerativeModel client in self.client (if not provided)
            - Calls _auto_fetch_model_info() to populate model metadata

        Raises:
            ImportError: If google-generativeai package is not installed.
            ValueError: If no API key is found in config or environment.
        """
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
        """Generate a completion using the Google Gemini API.

        Sends a prompt to the Gemini API and returns the generated response.
        Supports both single text prompts and chat-style message lists.
        When streaming is enabled, returns an iterator for processing chunks.

        Args:
            prompt: The input for generation. Can be either:
                - A string containing the prompt text
                - A list of message dictionaries with 'role' and 'content' keys
                  (will be formatted using _format_messages_for_gemini)
            model: Optional model override. If different from config.model,
                a new GenerativeModel client is created for this request.
            temperature: Sampling temperature override (0.0 to 1.0). Higher
                values produce more random output.
            max_tokens: Maximum number of tokens to generate in the response.
            top_p: Nucleus sampling parameter override (0.0 to 1.0).
            stop: List of sequences that will stop generation when encountered.
            stream: Whether to stream the response. If True, returns a
                streaming response iterator instead of a complete response.
            **kwargs: Additional Gemini-specific parameters passed directly
                to the generate_content method (e.g., safety_settings).

        Returns:
            If stream=False: A GenerateContentResponse object containing
                the complete generated text and metadata.
            If stream=True: A streaming response iterator that yields
                chunks as they are generated.

        Raises:
            RuntimeError: If the Gemini API request fails for any reason,
                wrapping the original exception with context.

        Example:
            # Simple text completion
            response = await llm.generate_completion("Explain photosynthesis")
            text = llm.extract_content(response)

            # Chat-style with messages
            messages = [
                {"role": "user", "content": "What's 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "And 3+3?"},
            ]
            response = await llm.generate_completion(messages)

            # Streaming response
            response = await llm.generate_completion("Write a poem", stream=True)
            async for chunk in response:
                print(chunk.text, end="")
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
        """Format chat-style messages into a Gemini-compatible prompt string.

        Converts a list of message dictionaries (OpenAI-style chat format)
        into a single formatted string suitable for Gemini's generate_content
        method. Each message is prefixed with its role for context.

        Note:
            Gemini's native API supports structured multi-turn conversations,
            but this implementation uses string concatenation for simplicity
            and compatibility with the standard message format used across
            all Xerxes LLM providers.

        Args:
            messages: List of message dictionaries, each containing:
                - 'role': The message role ('system', 'user', or 'assistant')
                - 'content': The message text content

        Returns:
            A formatted string with all messages concatenated, separated by
            double newlines. Each message is prefixed with its role:
            - "System: {content}" for system messages
            - "User: {content}" for user messages
            - "Assistant: {content}" for assistant messages
            - Raw content for unknown roles

        Example:
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ]
            formatted = llm._format_messages_for_gemini(messages)
            # Returns: "System: You are helpful.\\n\\nUser: Hello!"
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
        """Extract the text content from a Gemini API response.

        Parses the GenerateContentResponse object to extract the generated
        text. Handles multiple response formats including direct text access
        and candidate-based structures.

        The method attempts extraction in the following order:
        1. Direct .text attribute (simplest case)
        2. First candidate's content parts (structured response)

        Args:
            response: A GenerateContentResponse object from the Gemini API.
                Can be either a complete response or a streaming chunk.

        Returns:
            The extracted text content as a string. Returns an empty string
            if no text content is found or the response structure is
            unexpected.

        Example:
            response = await llm.generate_completion("Hello")
            text = llm.extract_content(response)
            print(text)  # "Hello! How can I help you today?"
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
        """Process a streaming response from Gemini with callback support.

        Iterates through the streaming response chunks from the Gemini API,
        extracting text content from each chunk and invoking the provided
        callback function. Accumulates and returns the complete response.

        This method is useful for real-time display of generated content
        or for implementing progress indicators during long generations.

        Args:
            response: A streaming response iterator from Gemini's
                generate_content method (called with stream=True).
            callback: A function called for each chunk with two arguments:
                - content (str): The text content in the current chunk
                - chunk (Any): The raw chunk object from Gemini API

        Returns:
            The complete accumulated text content from all chunks
            concatenated together.

        Example:
            def on_chunk(content: str, chunk: Any) -> None:
                print(content, end="", flush=True)

            response = await llm.generate_completion("Tell a story", stream=True)
            full_text = await llm.process_streaming_response(response, on_chunk)
            print(f"\\nTotal length: {len(full_text)}")
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
        """Stream completion chunks with function call detection.

        Processes a synchronous streaming response from the Gemini API,
        yielding standardized chunk dictionaries compatible with the
        Xerxes agent framework. Tracks accumulated content and provides
        metadata for each chunk.

        This method is used internally by agents to process streaming
        responses while detecting potential function calls in the output.

        Args:
            response: A synchronous streaming response iterator from
                Gemini's generate_content method (stream=True).
            agent: Optional agent instance for function call detection.
                Currently not used in this implementation but provided
                for interface compatibility.

        Yields:
            Dictionary containing streaming chunk information:
                - content (str | None): Text content in this chunk
                - buffered_content (str): Accumulated content so far
                - function_calls (list): Detected function calls (empty)
                - tool_calls (Any): Raw tool call data (None for Gemini)
                - raw_chunk (Any): The original Gemini chunk object
                - is_final (bool): Whether this is the final chunk

        Example:
            response = await llm.generate_completion("Hello", stream=True)
            for chunk in llm.stream_completion(response):
                if chunk["content"]:
                    print(chunk["content"], end="")
                if chunk["is_final"]:
                    print("\\n--- Generation complete ---")
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

            yield chunk_data

        yield {
            "content": None,
            "buffered_content": buffered_content,
            "function_calls": [],
            "tool_calls": None,
            "raw_chunk": None,
            "is_final": True,
        }

    async def astream_completion(
        self,
        response: Any,
        agent: Any | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Asynchronously stream completion chunks with function call detection.

        Processes an asynchronous streaming response from the Gemini API,
        yielding standardized chunk dictionaries compatible with the
        Xerxes agent framework. This is the async counterpart to
        stream_completion().

        This method enables non-blocking streaming of responses, allowing
        other async operations to proceed while waiting for chunks.

        Args:
            response: An asynchronous streaming response iterator from
                Gemini's async generate_content method.
            agent: Optional agent instance for function call detection.
                Currently not used in this implementation but provided
                for interface compatibility.

        Yields:
            Dictionary containing streaming chunk information:
                - content (str | None): Text content in this chunk
                - buffered_content (str): Accumulated content so far
                - function_calls (list): Detected function calls (empty)
                - tool_calls (Any): Raw tool call data (None for Gemini)
                - raw_chunk (Any): The original Gemini chunk object
                - is_final (bool): Whether this is the final chunk

        Example:
            response = await llm.generate_completion("Hello", stream=True)
            async for chunk in llm.astream_completion(response):
                if chunk["content"]:
                    print(chunk["content"], end="", flush=True)
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

        yield {
            "content": None,
            "buffered_content": buffered_content,
            "function_calls": [],
            "tool_calls": None,
            "raw_chunk": None,
            "is_final": True,
        }

    def parse_tool_calls(self, raw_data: Any) -> list[dict[str, Any]]:
        """Parse tool/function calls from Gemini response format.

        Extracts function call information from a Gemini API response
        and converts it to the standardized Xerxes tool call format.
        Gemini function calls are embedded within content parts of
        response candidates.

        This method enables agentic workflows where the model can request
        execution of predefined functions and receive their results.

        Args:
            raw_data: A GenerateContentResponse object from the Gemini API
                that may contain function call requests in its candidates.

        Returns:
            A list of tool call dictionaries, each containing:
                - id (str | None): Function call identifier (if available)
                - name (str): Name of the function to call
                - arguments (str): String representation of function arguments

            Returns an empty list if no function calls are found.

        Example:
            response = await llm.generate_completion(prompt, tools=tools)
            tool_calls = llm.parse_tool_calls(response)
            for call in tool_calls:
                result = execute_function(call["name"], call["arguments"])
        """

        tool_calls = []
        if hasattr(raw_data, "candidates") and raw_data.candidates:
            for candidate in raw_data.candidates:
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call"):
                            fc = part.function_call
                            import json as _json

                            args = getattr(fc, "args", None)
                            if args is not None:
                                try:
                                    args_str = _json.dumps(dict(args))
                                except (TypeError, ValueError):
                                    args_str = str(args)
                            else:
                                args_str = "{}"
                            tool_calls.append(
                                {
                                    "id": getattr(fc, "id", None),
                                    "name": fc.name,
                                    "arguments": args_str,
                                }
                            )
        return tool_calls

    def fetch_model_info(self) -> dict[str, Any]:
        """Fetch model metadata from the Gemini API.

        Retrieves information about the configured model from Google's
        model registry, including token limits and capabilities. This
        information is used to optimize token usage and prevent context
        overflow errors.

        The method is called automatically during client initialization
        via _auto_fetch_model_info() to populate config.max_model_len
        and config.model_metadata.

        Returns:
            A dictionary containing model metadata:
                - max_model_len (int | None): Maximum input tokens accepted
                - output_token_limit (int | None): Maximum output tokens

            Returns an empty dictionary if the model info cannot be
            fetched (e.g., network error, invalid model name).

        Note:
            This method silently catches exceptions to prevent
            initialization failures when model info is unavailable.
        """
        try:
            model_info = self.genai.get_model(f"models/{self.config.model}")
            return {
                "max_model_len": getattr(model_info, "input_token_limit", None),
                "output_token_limit": getattr(model_info, "output_token_limit", None),
            }
        except Exception:
            pass
        return {}
