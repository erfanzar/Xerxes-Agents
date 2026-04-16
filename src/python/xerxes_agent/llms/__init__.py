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


"""LLM providers for Xerxes.

This package provides a unified interface for integrating multiple Large Language
Model (LLM) providers into the Xerxes framework. Each provider implements the
:class:`BaseLLM` abstract interface, allowing seamless switching between backends.

The package includes built-in support for:
- **OpenAI**: GPT-4, GPT-4o, GPT-3.5 Turbo via :class:`OpenAILLM`
- **Anthropic**: Claude 3.x and 4.x models via :class:`AnthropicLLM`
- **Google Gemini**: Gemini Pro and Flash models via :class:`GeminiLLM`
- **Ollama**: Locally-hosted models (Llama, Mistral, etc.) via :class:`OllamaLLM`
- **Plugin providers**: Third-party providers registered via :class:`PluginRegistry`

Use the :func:`create_llm` factory function for convenient provider instantiation,
or import individual provider classes directly.

Example:
    >>> from xerxes_agent.llms import create_llm, LLMConfig
    >>> llm = create_llm("openai", model="gpt-4o", api_key="sk-...")
    >>> response = await llm.generate_completion("Hello!")
"""

from typing import Literal

from .anthropic import AnthropicLLM
from .base import BaseLLM, LLMConfig
from .compat import (
    CustomLLM,
    DeepSeekLLM,
    KimiLLM,
    LMStudioLLM,
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

try:
    from ..extensions.plugins import PluginRegistry
except ImportError:  # pragma: no cover - import guard for partial environments
    PluginRegistry = None  # type: ignore[assignment]


def _instantiate_provider(provider_impl, config: LLMConfig | None, kwargs: dict) -> BaseLLM:
    """Instantiate a provider implementation, class, or callable factory.

    Resolves a provider reference into a concrete :class:`BaseLLM` instance. The
    resolution strategy depends on the type of ``provider_impl``:

    1. **Already an instance**: If ``provider_impl`` is a :class:`BaseLLM` instance,
       it is returned directly.
    2. **A BaseLLM subclass**: If ``provider_impl`` is a class that extends
       :class:`BaseLLM`, it is instantiated with the given ``config`` (with
       ``kwargs`` merged into the config attributes) or with ``kwargs`` alone.
    3. **A callable factory**: If ``provider_impl`` is any other callable, it is
       called with ``config`` and ``kwargs``. Falls back to calling with ``kwargs``
       only if a ``TypeError`` is raised.

    Args:
        provider_impl: The provider to instantiate. Can be a :class:`BaseLLM`
            instance, a :class:`BaseLLM` subclass, or a callable that returns
            a :class:`BaseLLM` instance.
        config: Optional :class:`LLMConfig` to pass to the provider. When
            provided alongside a :class:`BaseLLM` subclass, ``kwargs`` are
            merged into the config's attributes before instantiation.
        kwargs: Additional keyword arguments for provider construction. These
            are set as attributes on ``config`` (if matching) or passed directly
            to the provider constructor.

    Returns:
        A fully initialized :class:`BaseLLM` instance ready for use.

    Raises:
        ValueError: If ``provider_impl`` is not a supported type (not a
            :class:`BaseLLM` instance, subclass, or callable).
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
    provider: Literal[
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
        "custom",
    ]
    | str,
    config: LLMConfig | None = None,
    plugin_registry: PluginRegistry | None = None,
    **kwargs,
) -> BaseLLM:
    """Factory function to create an LLM provider instance by name.

    Provides a convenient way to instantiate any supported LLM provider without
    importing the specific class. Supports both built-in providers and
    plugin-registered custom providers.

    The provider name is case-insensitive, and several aliases are supported
    (e.g., "claude" maps to :class:`AnthropicLLM`, "google" maps to
    :class:`GeminiLLM`).

    When a ``plugin_registry`` is provided, it is checked first for a matching
    provider before falling back to the built-in provider map. This allows
    third-party plugins to override or extend the available providers.

    Args:
        provider: The LLM provider name. Built-in options include:
            - ``"openai"``: :class:`OpenAILLM`
            - ``"anthropic"`` or ``"claude"``: :class:`AnthropicLLM`
            - ``"gemini"`` or ``"google"``: :class:`GeminiLLM`
            - ``"ollama"``: :class:`OllamaLLM`
            - ``"local"``: :class:`LocalLLM` (alias for OllamaLLM)
            Custom string names are also accepted when registered via plugins.
        config: Optional :class:`LLMConfig` object with provider configuration.
            If None, a config is created from ``kwargs`` by the provider class.
        plugin_registry: Optional :class:`PluginRegistry` instance to look up
            custom provider implementations before checking built-in providers.
            When a plugin provides the requested provider, it takes precedence.
        **kwargs: Additional provider-specific arguments passed to the provider
            constructor. Common kwargs include ``model``, ``api_key``,
            ``temperature``, ``max_tokens``, and ``base_url``.

    Returns:
        A fully initialized :class:`BaseLLM` instance for the requested provider.

    Raises:
        ValueError: If the provider name is not recognized and no matching
            plugin provider is found. The error message includes a list of
            all available built-in provider names.

    Example:
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
        "deepseek": DeepSeekLLM,
        "kimi": KimiLLM,
        "moonshot": KimiLLM,
        "qwen": QwenLLM,
        "dashscope": QwenLLM,
        "zhipu": ZhipuLLM,
        "glm": ZhipuLLM,
        "lmstudio": LMStudioLLM,
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
