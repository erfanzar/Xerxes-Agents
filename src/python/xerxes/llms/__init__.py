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
"""Init module for Xerxes.

Exports:
    - create_llm"""

from typing import Any, Literal

from .anthropic import AnthropicLLM
from .base import BaseLLM, LLMConfig
from .compat import (
    CustomLLM,
    DeepSeekLLM,
    KimiLLM,
    LMStudioLLM,
    MiniMaxLLM,
    OpenAICompatLLM,
    QwenLLM,
    ZhipuLLM,
)
from .gemini import GeminiLLM
from .ollama import LocalLLM, OllamaLLM
from .openai import OpenAILLM
from .registry import (
    COSTS,
    PROVIDERS,
    ProviderConfig,
    bare_model,
    calc_cost,
    detect_provider,
    get_api_key,
    get_context_limit,
    get_provider_config,
    list_all_models,
)

PluginRegistry: type[Any] | None = None
try:
    from ..extensions.plugins import PluginRegistry
except ImportError:
    pass


def _instantiate_provider(provider_impl, config: LLMConfig | None, kwargs: dict) -> BaseLLM:
    """Instantiate a provider class or return an existing instance.

    Handles BaseLLM instances directly, classes deriving from BaseLLM,
    and callable providers that accept config and kwargs.

    Args:
        provider_impl: Provider class, instance, or factory callable.
        config: Optional LLM configuration.
        kwargs: Additional keyword arguments passed to the provider.

    Returns:
        An initialized BaseLLM instance.

    Raises:
        ValueError: If the provider implementation type is unsupported.
    """
    if isinstance(provider_impl, BaseLLM):
        return provider_impl

    if isinstance(provider_impl, type) and issubclass(provider_impl, BaseLLM):
        if config:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            return provider_impl(config=config)
        return provider_impl(config=None, **kwargs)

    if callable(provider_impl):
        if config:
            try:
                return provider_impl(config=config, **kwargs)
            except TypeError:
                return provider_impl(**kwargs)
        try:
            return provider_impl(config=None, **kwargs)
        except TypeError:
            return provider_impl(**kwargs)

    raise ValueError(f"Unsupported provider implementation: {provider_impl!r}")


def create_llm(
    provider: (
        Literal[
            "openai",
            "anthropic",
            "claude",
            "gemini",
            "google",
            "ollama",
            "local",
            "deepseek",
            "kimi",
            "moonshot",
            "qwen",
            "dashscope",
            "zhipu",
            "glm",
            "lmstudio",
            "minimax",
            "custom",
        ]
        | str
    ),
    config: LLMConfig | None = None,
    plugin_registry: Any | None = None,
    **kwargs,
) -> BaseLLM:
    """Factory function that creates an LLM client for the specified provider.

    Routes the provider name to the appropriate LLM implementation class,
    applies configuration overrides from kwargs, and delegates to a plugin
    registry if one is provided.

    Args:
        provider: The name of the LLM provider (e.g., "openai", "anthropic").
        config: Optional shared LLM configuration applied before kwargs.
        plugin_registry: Optional plugin registry consulted before built-in providers.
        **kwargs: Additional parameters forwarded to the LLM constructor
            (e.g., model, temperature, api_key).

    Returns:
        An initialized LLM client instance.

    Raises:
        ValueError: If the provider name is not recognized.
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
        "deepseek": DeepSeekLLM,
        "kimi": KimiLLM,
        "moonshot": KimiLLM,
        "qwen": QwenLLM,
        "dashscope": QwenLLM,
        "zhipu": ZhipuLLM,
        "glm": ZhipuLLM,
        "lmstudio": LMStudioLLM,
        "minimax": MiniMaxLLM,
        "custom": CustomLLM,
    }

    if plugin_registry is not None:
        plugin_provider = plugin_registry.get_provider(provider)
        if plugin_provider is not None:
            return _instantiate_provider(plugin_provider, config, kwargs)

    if provider not in providers:
        available = ", ".join(providers.keys())
        raise ValueError(f"Unknown LLM provider: {provider}. Available: {available}")

    return _instantiate_provider(providers[provider], config, kwargs)


__all__ = [
    "COSTS",
    "PROVIDERS",
    "AnthropicLLM",
    "BaseLLM",
    "CustomLLM",
    "DeepSeekLLM",
    "GeminiLLM",
    "KimiLLM",
    "LLMConfig",
    "LMStudioLLM",
    "LocalLLM",
    "MiniMaxLLM",
    "OllamaLLM",
    "OpenAICompatLLM",
    "OpenAILLM",
    "ProviderConfig",
    "QwenLLM",
    "ZhipuLLM",
    "bare_model",
    "calc_cost",
    "create_llm",
    "detect_provider",
    "get_api_key",
    "get_context_limit",
    "get_provider_config",
    "list_all_models",
]
