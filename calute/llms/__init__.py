
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

"""LLM providers for Calute."""

from typing import Literal

from .anthropic import AnthropicLLM
from .base import BaseLLM, LLMConfig
from .gemini import GeminiLLM
from .ollama import LocalLLM, OllamaLLM
from .openai import OpenAILLM


def create_llm(
    provider: Literal["openai", "anthropic", "claude", "gemini", "google", "ollama", "local"] | str,
    config: LLMConfig | None = None,
    **kwargs,
) -> BaseLLM:
    """Factory function to create an LLM instance.

    Args:
        provider: The LLM provider name (openai, anthropic, gemini, ollama, etc.)
        config: Optional LLMConfig object
        **kwargs: Additional provider-specific arguments

    Returns:
        An instance of the appropriate LLM class

    Raises:
        ValueError: If the provider is not supported

    Examples:
        >>> llm = create_llm("openai", model="gpt-4")
        >>> llm = create_llm("anthropic", api_key="...", model="claude-3-opus")
        >>> llm = create_llm("ollama", base_url="http://localhost:11434", model="llama2")
    """
    provider = provider.lower()

    providers = {
        "openai": OpenAILLM,
        "anthropic": AnthropicLLM,
        "claude": AnthropicLLM,
        "gemini": GeminiLLM,
        "google": GeminiLLM,
        "ollama": OllamaLLM,
        "local": LocalLLM,
    }

    if provider not in providers:
        available = ", ".join(providers.keys())
        raise ValueError(f"Unknown LLM provider: {provider}. Available: {available}")

    llm_class = providers[provider]

    if config:
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return llm_class(config=config)
    else:
        return llm_class(config=None, **kwargs)


__all__ = [
    "AnthropicLLM",
    "BaseLLM",
    "GeminiLLM",
    "LLMConfig",
    "LocalLLM",
    "OllamaLLM",
    "OpenAILLM",
    "create_llm",
]
