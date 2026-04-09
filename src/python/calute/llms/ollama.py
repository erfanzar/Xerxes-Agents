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


"""Ollama local LLM provider implementation.

This module provides integration with Ollama for running large language models
locally. Ollama is an open-source tool for running LLMs on your own machine,
supporting models like Llama, Mistral, CodeLlama, and many others.

The module handles:
- HTTP communication with the Ollama server via async httpx
- Both chat-style (/api/chat) and generate-style (/api/generate) endpoints
- Streaming response processing with callback support
- Automatic model metadata fetching (context length, parameters)
- Configurable timeout for long-running generations

Supported models include:
- llama2, llama3 (default: llama2)
- mistral, mixtral
- codellama
- phi, phi3
- Any model available in your local Ollama installation

Typical usage example:
    from calute.llms.ollama import OllamaLLM
    from calute.llms.base import LLMConfig

    # Ensure Ollama is running locally (ollama serve)
    config = LLMConfig(
        model="llama3",
        base_url="http://localhost:11434",
        temperature=0.7,
        max_tokens=2048,
    )

    async with OllamaLLM(config) as llm:
        response = await llm.generate_completion("Explain recursion")
        content = llm.extract_content(response)
        print(content)

Note:
    Requires the httpx package to be installed:
    pip install httpx

    Also requires Ollama to be installed and running:
    https://ollama.ai
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from .base import BaseLLM, LLMConfig

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    httpx = None


class OllamaLLM(BaseLLM):
    """Ollama local LLM provider implementation.

    OllamaLLM provides integration with the Ollama server for running
    language models locally. It communicates with Ollama via HTTP,
    supporting both text generation and chat-style conversations.

    This implementation automatically selects the appropriate Ollama
    endpoint based on the prompt format:
    - String prompts use /api/generate
    - Message lists use /api/chat

    Attributes:
        config: LLMConfig instance containing provider configuration.
        client: httpx.AsyncClient instance for HTTP communication with
            the Ollama server.

    Example:
        # Basic usage with string prompt
        llm = OllamaLLM(model="llama3", base_url="http://localhost:11434")
        response = await llm.generate_completion("What is Python?")
        print(llm.extract_content(response))

        # Using with chat-style messages
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a hello world in Python"},
        ]
        response = await llm.generate_completion(messages)

        # Streaming response
        response = await llm.generate_completion("Tell me a joke", stream=True)
        async for chunk in response:
            print(llm.extract_content(chunk), end="")

    Note:
        Ollama must be running locally (or remotely) before using this class.
        Start Ollama with: ollama serve
    """

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize the Ollama LLM provider.

        Creates a new OllamaLLM instance with the specified configuration.
        If no config is provided, a default configuration is created using
        sensible defaults for local Ollama deployment.

        Args:
            config: LLM configuration object. If None, a default config is
                created with:
                - model: "llama2" (or from kwargs)
                - base_url: "http://localhost:11434" (or from kwargs)
                - timeout: 120.0 seconds (or from kwargs)
            **kwargs: Additional configuration parameters passed to LLMConfig
                when config is None. Common kwargs include:
                - model: Model name (e.g., "llama3", "mistral")
                - base_url: Ollama server URL
                - timeout: Request timeout in seconds
                - temperature: Sampling temperature

        Raises:
            ImportError: If httpx package is not installed.

        Example:
            # Using explicit config
            config = LLMConfig(
                model="codellama",
                base_url="http://localhost:11434",
                temperature=0.5,
            )
            llm = OllamaLLM(config)

            # Using kwargs for convenience
            llm = OllamaLLM(model="mistral", temperature=0.7)

            # Remote Ollama server
            llm = OllamaLLM(
                model="llama3",
                base_url="http://remote-server:11434"
            )
        """
        if not HAS_HTTPX:
            raise ImportError("httpx library required for Ollama. Install with: pip install httpx")

        if config is None:
            config = LLMConfig(
                model=kwargs.pop("model", "llama2"),
                base_url=kwargs.pop("base_url", "http://localhost:11434"),
                timeout=kwargs.pop("timeout", 120.0),
                **kwargs,
            )

        self.client = None
        super().__init__(config)

    def _initialize_client(self) -> None:
        """Initialize the async HTTP client for Ollama communication.

        Creates an httpx.AsyncClient configured with the base URL and
        timeout from the provider configuration. This method is called
        automatically at the end of __init__.

        Side Effects:
            - Creates httpx.AsyncClient in self.client
            - Calls _auto_fetch_model_info() to populate model metadata

        Note:
            The client uses the Ollama server URL from config.base_url
            (defaults to http://localhost:11434) and config.timeout
            (defaults to 120 seconds for long generations).
        """
        self.client = httpx.AsyncClient(
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
        **kwargs,
    ) -> Any:
        """Generate a completion using the Ollama API.

        Sends a prompt to the local Ollama server and returns the generated
        response. Automatically selects the appropriate API endpoint based
        on the prompt format:
        - String prompts → /api/generate
        - Message lists → /api/chat

        Args:
            prompt: The input for generation. Can be either:
                - A string containing the prompt text
                - A list of message dictionaries with 'role' and 'content'
                  keys for chat-style conversations
            model: Optional model override. Uses config.model if not specified.
            temperature: Sampling temperature override. Controls randomness
                in the output (higher = more random).
            max_tokens: Maximum number of tokens to generate (maps to
                Ollama's num_predict option).
            top_p: Nucleus sampling parameter override (0.0 to 1.0).
            stop: List of sequences that will stop generation when encountered.
            stream: Whether to stream the response. If True, returns an
                async iterator yielding chunks as they are generated.
            **kwargs: Additional Ollama-specific parameters. Supports:
                - options: Dict of model options (merged with defaults)
                - Any other parameters passed directly to the API

        Returns:
            If stream=False: A dictionary containing the complete response
                with 'response' key (for /api/generate) or 'message' key
                (for /api/chat).
            If stream=True: An async iterator yielding response chunks
                as dictionaries.

        Raises:
            RuntimeError: If the Ollama API request fails (server error,
                timeout, connection refused, etc.).

        Example:
            # Simple text completion
            response = await llm.generate_completion("Write a haiku about coding")
            text = llm.extract_content(response)

            # Chat-style with messages
            messages = [
                {"role": "system", "content": "You are a poet."},
                {"role": "user", "content": "Write about the moon."},
            ]
            response = await llm.generate_completion(messages)

            # Streaming response
            response = await llm.generate_completion("Tell a story", stream=True)
            async for chunk in response:
                print(llm.extract_content(chunk), end="", flush=True)

            # With custom Ollama options
            response = await llm.generate_completion(
                "Hello",
                options={"seed": 42, "repeat_penalty": 1.1}
            )
        """
        use_stream = stream if stream is not None else self.config.stream

        if isinstance(prompt, list):
            endpoint = "/api/chat"
            payload = {
                "model": model or self.config.model,
                "messages": prompt,
                "stream": use_stream,
                "options": {
                    "temperature": temperature if temperature is not None else self.config.temperature,
                    "top_p": top_p if top_p is not None else self.config.top_p,
                    "num_predict": max_tokens if max_tokens is not None else self.config.max_tokens,
                },
            }
        else:
            endpoint = "/api/generate"
            payload = {
                "model": model or self.config.model,
                "prompt": prompt,
                "stream": use_stream,
                "options": {
                    "temperature": temperature if temperature is not None else self.config.temperature,
                    "top_p": top_p if top_p is not None else self.config.top_p,
                    "num_predict": max_tokens if max_tokens is not None else self.config.max_tokens,
                },
            }

        if stop or self.config.stop:
            payload["options"]["stop"] = stop or self.config.stop

        if self.config.top_k:
            payload["options"]["top_k"] = self.config.top_k

        if "options" in kwargs:
            payload["options"].update(kwargs.pop("options"))
        payload.update(kwargs)

        try:
            if use_stream:
                return self._stream_completion(endpoint, payload)
            else:
                response = await self.client.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama API request failed: {e}") from e

    async def _stream_completion(self, endpoint: str, payload: dict) -> AsyncIterator[dict]:
        """Stream completion responses from the Ollama API.

        Internal method that handles streaming HTTP requests to the Ollama
        server. Parses the newline-delimited JSON (NDJSON) response format
        used by Ollama's streaming endpoints.

        Args:
            endpoint: The Ollama API endpoint path (e.g., "/api/generate"
                or "/api/chat").
            payload: The request payload dictionary to send as JSON body,
                containing model, prompt/messages, and options.

        Yields:
            Dictionaries parsed from each line of the streaming response.
            For /api/generate: {"response": "text", "done": bool, ...}
            For /api/chat: {"message": {"content": "text"}, "done": bool, ...}

        Raises:
            httpx.HTTPStatusError: If the server returns an error status code.
            httpx.HTTPError: If the request fails for network reasons.

        Note:
            This method uses httpx's async streaming context manager to
            efficiently process large responses without buffering the
            entire response in memory.
        """
        async with self.client.stream("POST", endpoint, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    yield json.loads(line)

    def extract_content(self, response: Any) -> str:
        """Extract the text content from an Ollama API response.

        Parses the response dictionary to extract the generated text.
        Handles both endpoint formats:
        - /api/chat responses have {"message": {"content": "text"}}
        - /api/generate responses have {"response": "text"}

        Args:
            response: A dictionary from the Ollama API. Can be either
                a complete response or a streaming chunk.

        Returns:
            The extracted text content as a string. Returns an empty string
            if the response format is unexpected or contains no content.

        Example:
            response = await llm.generate_completion("Hello")
            text = llm.extract_content(response)
            print(text)  # "Hello! How can I help you today?"
        """
        if isinstance(response, dict):
            if "message" in response:
                return response["message"].get("content", "")

            return response.get("response", "")
        return ""

    async def process_streaming_response(
        self,
        response: Any,
        callback: Callable[[str, Any], None],
    ) -> str:
        """Process a streaming response from Ollama with callback support.

        Iterates through the streaming response chunks from the Ollama API,
        extracting text content from each chunk and invoking the provided
        callback function. Accumulates and returns the complete response.

        This method is useful for real-time display of generated content
        or for implementing progress indicators during long generations.

        Args:
            response: An async iterator yielding response chunk dictionaries
                from _stream_completion().
            callback: A function called for each chunk containing text,
                with two arguments:
                - content (str): The text content in the current chunk
                - chunk (dict): The raw chunk dictionary from Ollama API

        Returns:
            The complete accumulated text content from all chunks
            concatenated together.

        Example:
            def on_chunk(content: str, chunk: dict) -> None:
                print(content, end="", flush=True)

            response = await llm.generate_completion("Tell a story", stream=True)
            full_text = await llm.process_streaming_response(response, on_chunk)
            print(f"\\nTotal length: {len(full_text)}")
        """
        accumulated_content = ""

        async for chunk in response:
            if isinstance(chunk, dict):
                if "message" in chunk:
                    content = chunk["message"].get("content", "")

                else:
                    content = chunk.get("response", "")

                if content:
                    accumulated_content += content
                    callback(content, chunk)

        return accumulated_content

    def stream_completion(
        self,
        response: Any,
        agent: Any | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream completion chunks with function call detection.

        Processes a synchronous streaming response from the Ollama API,
        yielding standardized chunk dictionaries compatible with the
        Calute agent framework. Tracks accumulated content and detects
        the final chunk via Ollama's "done" flag.

        This method is used internally by agents to process streaming
        responses while maintaining compatibility with other LLM providers.

        Args:
            response: A synchronous iterator yielding response chunk
                dictionaries from the Ollama streaming API.
            agent: Optional agent instance for function call detection.
                Currently not used as Ollama has limited function calling
                support, but provided for interface compatibility.

        Yields:
            Dictionary containing streaming chunk information:
                - content (str | None): Text content in this chunk
                - buffered_content (str): Accumulated content so far
                - function_calls (list): Detected function calls (empty)
                - tool_calls (Any): Raw tool call data (None)
                - raw_chunk (dict): The original Ollama chunk dictionary
                - is_final (bool): True when Ollama's "done" flag is True

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

            if isinstance(chunk, dict):
                if "message" in chunk:
                    content = chunk["message"].get("content", "")
                else:
                    content = chunk.get("response", "")

                if content:
                    buffered_content += content
                    chunk_data["content"] = content
                    chunk_data["buffered_content"] = buffered_content

                if chunk.get("done", False):
                    chunk_data["is_final"] = True

            yield chunk_data

    async def astream_completion(
        self,
        response: Any,
        agent: Any | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Asynchronously stream completion chunks with function call detection.

        Processes an asynchronous streaming response from the Ollama API,
        yielding standardized chunk dictionaries compatible with the
        Calute agent framework. This is the async counterpart to
        stream_completion().

        This method enables non-blocking streaming of responses, allowing
        other async operations to proceed while waiting for chunks from
        the Ollama server.

        Args:
            response: An asynchronous iterator yielding response chunk
                dictionaries from _stream_completion().
            agent: Optional agent instance for function call detection.
                Currently not used as Ollama has limited function calling
                support, but provided for interface compatibility.

        Yields:
            Dictionary containing streaming chunk information:
                - content (str | None): Text content in this chunk
                - buffered_content (str): Accumulated content so far
                - function_calls (list): Detected function calls (empty)
                - tool_calls (Any): Raw tool call data (None)
                - raw_chunk (dict): The original Ollama chunk dictionary
                - is_final (bool): True when Ollama's "done" flag is True

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

            if isinstance(chunk, dict):
                if "message" in chunk:
                    content = chunk["message"].get("content", "")
                else:
                    content = chunk.get("response", "")

                if content:
                    buffered_content += content
                    chunk_data["content"] = content
                    chunk_data["buffered_content"] = buffered_content

                if chunk.get("done", False):
                    chunk_data["is_final"] = True

            yield chunk_data

    def parse_tool_calls(self, raw_data: Any) -> list[dict[str, Any]]:
        """Parse tool/function calls from Ollama response format.

        Placeholder method for tool call parsing. Ollama has limited
        native support for function calling compared to cloud providers,
        so this method currently returns an empty list.

        Note:
            Some Ollama models support function calling through special
            prompting or the tools parameter, but the response format
            varies by model. Future versions may implement parsing for
            models that support structured function calling.

        Args:
            raw_data: Response data from Ollama API that may contain
                tool call information (currently unused).

        Returns:
            An empty list, as Ollama function calling support is limited.
        """
        return []

    def fetch_model_info(self) -> dict[str, Any]:
        """Fetch model metadata from the Ollama /api/show endpoint.

        Queries the Ollama server for detailed information about the
        configured model, including context length, parameter count,
        model family, and quantization level.

        The method is called automatically during client initialization
        via _auto_fetch_model_info() to populate config.max_model_len
        and config.model_metadata.

        Returns:
            A dictionary containing model metadata:
                - max_model_len (int | None): Maximum context length in tokens
                - parameter_size (str | None): Model parameter count (e.g., "7B")
                - family (str | None): Model family (e.g., "llama")
                - quantization_level (str | None): Quantization type (e.g., "Q4_0")

            Returns an empty dictionary if the model info cannot be
            fetched (e.g., model not found, server not running).

        Note:
            This method uses a synchronous HTTP client with a short timeout
            to avoid blocking async initialization. Context length is
            looked up in multiple possible locations in the response.
        """
        try:
            with httpx.Client(base_url=self.config.base_url, timeout=10.0) as client:
                resp = client.post("/api/show", json={"name": self.config.model})
                if resp.status_code == 200:
                    data = resp.json()
                    model_info = data.get("model_info", {})
                    details = data.get("details", {})
                    context_len = (
                        model_info.get("context_length")
                        or model_info.get("llama.context_length")
                        or model_info.get("num_ctx")
                    )
                    return {
                        "max_model_len": context_len,
                        "parameter_size": details.get("parameter_size"),
                        "family": details.get("family"),
                        "quantization_level": details.get("quantization_level"),
                    }
        except Exception:
            pass
        return {}

    async def close(self) -> None:
        """Close the HTTP client and release resources.

        Properly closes the async HTTP client connection to the Ollama
        server. This method should be called when done using the LLM
        provider to ensure clean resource cleanup.

        This method is called automatically when using the provider
        as an async context manager:

        Example:
            async with OllamaLLM(model="llama3") as llm:
                response = await llm.generate_completion("Hello")
            # client is automatically closed here

        Note:
            Safe to call multiple times; will only close if client exists.
        """
        if self.client:
            await self.client.aclose()


class LocalLLM(OllamaLLM):
    """Alias for OllamaLLM for backward compatibility.

    LocalLLM is a convenience alias that points to OllamaLLM. It provides
    backward compatibility for code that used the LocalLLM name before
    the rename to the more specific OllamaLLM.

    All functionality is identical to OllamaLLM. New code should prefer
    using OllamaLLM directly.

    Example:
        # These are equivalent:
        llm1 = LocalLLM(model="llama3")
        llm2 = OllamaLLM(model="llama3")

    See Also:
        OllamaLLM: The primary class for Ollama integration.
    """
