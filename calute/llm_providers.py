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

"""Extended LLM provider support."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Callable
from typing import Any

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    httpx = None

try:
    from tenacity import retry
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False
    # Dummy decorator
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .client import LLMClient
from .errors import ClientError
from .logging_config import get_logger

logger = get_logger(__name__)


class AnthropicClient(LLMClient):
    """Anthropic Claude client implementation."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        version: str = "2023-06-01",
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")

        self.base_url = base_url
        self.version = version
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "anthropic-version": version,
                "x-api-key": self.api_key,
                "content-type": "application/json",
            },
            timeout=60.0,
        )

    async def generate_completion(
        self,
        prompt: str | list[dict[str, str]],
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        stream: bool = False,
        **kwargs,
    ) -> Any:
        """Generate completion using Anthropic Claude."""
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = self._convert_messages(prompt)

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        if stop:
            payload["stop_sequences"] = stop

        # Add any extra parameters
        payload.update(kwargs)

        try:
            if stream:
                return await self._stream_completion(payload)
            else:
                response = await self.client.post("/v1/messages", json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            raise ClientError("anthropic", f"Request failed: {e}", original_error=e) from e

    async def _stream_completion(self, payload: dict) -> AsyncIterator[dict]:
        """Stream completion from Anthropic."""
        payload["stream"] = True

        async with self.client.stream("POST", "/v1/messages", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data != "[DONE]":
                        yield json.loads(data)

    def _convert_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Convert OpenAI-style messages to Anthropic format."""
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Anthropic uses "user" and "assistant" roles
            if role == "system":
                # Prepend system message to first user message
                if converted and converted[0]["role"] == "user":
                    converted[0]["content"] = f"{content}\n\n{converted[0]['content']}"
                else:
                    converted.insert(0, {"role": "user", "content": content})
            else:
                converted.append({"role": role, "content": content})

        return converted

    def extract_content(self, response: Any) -> str:
        """Extract content from Anthropic response."""
        if isinstance(response, dict):
            return response.get("content", [{}])[0].get("text", "")
        return ""

    async def process_streaming_response(
        self,
        response: Any,
        callback: Callable[[str, Any], None],
    ) -> str:
        """Process streaming response from Anthropic."""
        buffered_content = ""

        async for chunk in response:
            if chunk.get("type") == "content_block_delta":
                delta = chunk.get("delta", {})
                if text := delta.get("text"):
                    buffered_content += text
                    callback(text, chunk)

        return buffered_content


class CohereClient(LLMClient):
    """Cohere client implementation."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.cohere.ai",
    ):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key not provided")

        self.base_url = base_url
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    async def generate_completion(
        self,
        prompt: str | list[dict[str, str]],
        model: str = "command-r-plus",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        stream: bool = False,
        **kwargs,
    ) -> Any:
        """Generate completion using Cohere."""
        if isinstance(prompt, list):
            # Convert messages to single prompt
            prompt = self._messages_to_prompt(prompt)

        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "p": top_p,
            "stream": stream,
        }

        if stop:
            payload["stop_sequences"] = stop

        payload.update(kwargs)

        try:
            if stream:
                return await self._stream_completion(payload)
            else:
                response = await self.client.post("/v1/generate", json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            raise ClientError("cohere", f"Request failed: {e}", original_error=e) from e

    async def _stream_completion(self, payload: dict) -> AsyncIterator[dict]:
        """Stream completion from Cohere."""
        async with self.client.stream("POST", "/v1/generate", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    yield json.loads(line)

    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        """Convert messages to a single prompt."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role.upper()}: {content}")
        return "\n\n".join(prompt_parts)

    def extract_content(self, response: Any) -> str:
        """Extract content from Cohere response."""
        if isinstance(response, dict):
            generations = response.get("generations", [])
            if generations:
                return generations[0].get("text", "")
        return ""

    async def process_streaming_response(
        self,
        response: Any,
        callback: Callable[[str, Any], None],
    ) -> str:
        """Process streaming response from Cohere."""
        buffered_content = ""

        async for chunk in response:
            if text := chunk.get("text"):
                buffered_content += text
                callback(text, chunk)

        return buffered_content


class HuggingFaceClient(LLMClient):
    """HuggingFace Inference API client."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api-inference.huggingface.co",
    ):
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("HuggingFace API key not provided")

        self.base_url = base_url
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,  # Longer timeout for model loading
        )

    async def generate_completion(
        self,
        prompt: str | list[dict[str, str]],
        model: str = "meta-llama/Llama-2-70b-chat-hf",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        stream: bool = False,
        **kwargs,
    ) -> Any:
        """Generate completion using HuggingFace."""
        if isinstance(prompt, list):
            prompt = self._format_chat_prompt(prompt, model)

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": True,
                "return_full_text": False,
            },
            "options": {
                "use_cache": False,
                "wait_for_model": True,
            },
        }

        if stop:
            payload["parameters"]["stop"] = stop

        # Merge additional parameters
        if "parameters" in kwargs:
            payload["parameters"].update(kwargs["parameters"])

        try:
            response = await self.client.post(
                f"/models/{model}",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise ClientError("huggingface", f"Request failed: {e}", original_error=e) from e

    def _format_chat_prompt(self, messages: list[dict[str, str]], model: str) -> str:
        """Format messages for chat models."""
        # Different models may require different formats
        if "llama" in model.lower():
            return self._format_llama_prompt(messages)
        elif "mistral" in model.lower():
            return self._format_mistral_prompt(messages)
        else:
            # Generic format
            return self._generic_chat_format(messages)

    def _format_llama_prompt(self, messages: list[dict[str, str]]) -> str:
        """Format for Llama models."""
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt += f"<<SYS>>\n{content}\n<</SYS>>\n\n"
            elif role == "user":
                prompt += f"[INST] {content} [/INST]\n"
            elif role == "assistant":
                prompt += f"{content}\n"

        return prompt

    def _format_mistral_prompt(self, messages: list[dict[str, str]]) -> str:
        """Format for Mistral models."""
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                prompt += f"[INST] {content} [/INST]\n"
            else:
                prompt += f"{content}\n"

        return prompt

    def _generic_chat_format(self, messages: list[dict[str, str]]) -> str:
        """Generic chat format."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        return "\n\n".join(prompt_parts)

    def extract_content(self, response: Any) -> str:
        """Extract content from HuggingFace response."""
        if isinstance(response, list) and response:
            return response[0].get("generated_text", "")
        elif isinstance(response, dict):
            return response.get("generated_text", "")
        return ""

    async def process_streaming_response(
        self,
        response: Any,
        callback: Callable[[str, Any], None],
    ) -> str:
        """HuggingFace doesn't support streaming in the same way."""
        # For HuggingFace, we'll just return the complete response
        content = self.extract_content(response)
        callback(content, response)
        return content


class LocalLLMClient(LLMClient):
    """Client for local LLM servers (Ollama, LM Studio, etc.)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",  # Ollama default
        api_type: str = "ollama",  # ollama, lm-studio, text-generation-webui
    ):
        self.base_url = base_url
        self.api_type = api_type
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=120.0,
        )

    async def generate_completion(
        self,
        prompt: str | list[dict[str, str]],
        model: str = "llama2",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        stream: bool = False,
        **kwargs,
    ) -> Any:
        """Generate completion using local LLM."""
        if self.api_type == "ollama":
            return await self._ollama_completion(
                prompt, model, temperature, max_tokens, top_p, stop, stream, **kwargs
            )
        elif self.api_type == "lm-studio":
            return await self._lm_studio_completion(
                prompt, model, temperature, max_tokens, top_p, stop, stream, **kwargs
            )
        else:
            raise ValueError(f"Unsupported local LLM type: {self.api_type}")

    async def _ollama_completion(
        self,
        prompt: str | list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop: list[str] | None,
        stream: bool,
        **kwargs,
    ) -> Any:
        """Generate completion using Ollama."""
        if isinstance(prompt, list):
            # Convert to Ollama chat format
            payload = {
                "model": model,
                "messages": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                },
            }
            endpoint = "/api/chat"
        else:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                },
            }
            endpoint = "/api/generate"

        if stop:
            payload["options"]["stop"] = stop

        try:
            if stream:
                return await self._stream_ollama(endpoint, payload)
            else:
                response = await self.client.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            raise ClientError("ollama", f"Request failed: {e}", original_error=e) from e

    async def _stream_ollama(self, endpoint: str, payload: dict) -> AsyncIterator[dict]:
        """Stream from Ollama."""
        async with self.client.stream("POST", endpoint, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    yield json.loads(line)

    async def _lm_studio_completion(
        self,
        prompt: str | list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop: list[str] | None,
        stream: bool,
        **kwargs,
    ) -> Any:
        """Generate completion using LM Studio (OpenAI-compatible)."""
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream,
        }

        if stop:
            payload["stop"] = stop

        try:
            response = await self.client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise ClientError("lm-studio", f"Request failed: {e}", original_error=e) from e

    def extract_content(self, response: Any) -> str:
        """Extract content from local LLM response."""
        if self.api_type == "ollama":
            if "message" in response:
                return response["message"].get("content", "")
            return response.get("response", "")
        elif self.api_type == "lm-studio":
            choices = response.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
        return ""

    async def process_streaming_response(
        self,
        response: Any,
        callback: Callable[[str, Any], None],
    ) -> str:
        """Process streaming response from local LLM."""
        buffered_content = ""

        async for chunk in response:
            if self.api_type == "ollama":
                if "message" in chunk:
                    content = chunk["message"].get("content", "")
                else:
                    content = chunk.get("response", "")
            else:
                # OpenAI-compatible format
                choices = chunk.get("choices", [])
                if choices:
                    content = choices[0].get("delta", {}).get("content", "")
                else:
                    content = ""

            if content:
                buffered_content += content
                callback(content, chunk)

        return buffered_content


# Factory function to create LLM clients
def create_llm_client(provider: str, **kwargs) -> LLMClient:
    """Create an LLM client based on the provider."""
    provider = provider.lower()

    if provider == "openai":
        from openai import OpenAI

        from .client import OpenAIClient
        client = OpenAI(api_key=kwargs.get("api_key"))
        return OpenAIClient(client)

    elif provider == "gemini":
        import google.generativeai as genai

        from .client import GeminiClient
        genai.configure(api_key=kwargs.get("api_key"))
        return GeminiClient(genai)

    elif provider == "anthropic":
        return AnthropicClient(**kwargs)

    elif provider == "cohere":
        return CohereClient(**kwargs)

    elif provider == "huggingface":
        return HuggingFaceClient(**kwargs)

    elif provider in ["ollama", "lm-studio", "local"]:
        return LocalLLMClient(
            base_url=kwargs.get("base_url", "http://localhost:11434"),
            api_type=kwargs.get("api_type", provider),
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
