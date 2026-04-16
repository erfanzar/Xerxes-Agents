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


"""OpenAI-compatible provider for non-native backends.

This module provides :class:`OpenAICompatLLM`, a thin wrapper around
:class:`OpenAILLM` that automatically configures the ``base_url`` and
``api_key`` for OpenAI-compatible providers such as:

- **DeepSeek**: ``deepseek-chat``, ``deepseek-reasoner``
- **Kimi / Moonshot**: ``moonshot-v1-8k``, ``kimi-latest``
- **Qwen / DashScope**: ``qwen-max``, ``qwen-plus``, ``qwen-turbo``
- **Zhipu / GLM**: ``glm-4-plus``, ``glm-4-flash``
- **LM Studio**: Any locally-loaded model
- **Custom**: Any OpenAI-compatible endpoint

The class reads provider metadata from :mod:`xerxes_agent.llms.registry` and
resolves API keys from environment variables automatically.

Usage::

    from xerxes_agent.llms.compat import OpenAICompatLLM

    # Auto-detect provider from model name
    llm = OpenAICompatLLM(model="deepseek-chat")

    # Explicit provider
    llm = OpenAICompatLLM(model="my-model", provider="custom",
                          base_url="http://localhost:8000/v1")

    # With LLMConfig
    from xerxes_agent.llms.base import LLMConfig
    config = LLMConfig(model="qwen-max")
    llm = OpenAICompatLLM(config=config, provider="qwen")
"""

from __future__ import annotations

from typing import Any

from .base import LLMConfig
from .openai import OpenAILLM
from .registry import (
    PROVIDERS,
    bare_model,
    detect_provider,
    get_api_key,
)


class OpenAICompatLLM(OpenAILLM):
    """OpenAI-compatible LLM provider for third-party backends.

    Extends :class:`OpenAILLM` by auto-resolving the ``base_url`` and
    ``api_key`` from the provider registry. The provider can be specified
    explicitly or inferred from the model name.

    This allows a single class to handle DeepSeek, Kimi, Qwen, Zhipu,
    LM Studio, and any custom OpenAI-compatible endpoint without
    requiring separate implementations.

    Attributes:
        provider_name: The resolved provider name (e.g. ``"deepseek"``).

    Example:
        >>> llm = OpenAICompatLLM(model="deepseek-chat")
        >>> llm.provider_name
        'deepseek'
        >>> response = await llm.generate_completion("Hello!")
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        provider: str | None = None,
        client: Any | None = None,
        async_client: Any | None = None,
        **kwargs,
    ):
        """Initialize the OpenAI-compatible LLM provider.

        Args:
            config: Optional LLMConfig. If ``base_url`` and ``api_key`` are
                not set, they are resolved from the provider registry.
            provider: Explicit provider name. If None, auto-detected from
                the model name via :func:`detect_provider`.
            client: Optional pre-configured OpenAI client.
            async_client: Optional pre-configured AsyncOpenAI client.
            **kwargs: Passed through to :class:`OpenAILLM`.
        """
        model = config.model if config else kwargs.get("model", "")
        self.provider_name = provider or detect_provider(model)

        prov = PROVIDERS.get(self.provider_name)

        if config is None:
            kwargs.setdefault("model", model)
            config = LLMConfig(**kwargs)

        config.model = bare_model(config.model)

        if not config.base_url and prov:
            config.base_url = prov.base_url

        if not config.api_key and prov:
            config.api_key = get_api_key(self.provider_name)

        if prov and not config.max_model_len:
            config.max_model_len = prov.context_limit

        super().__init__(config=config, client=client, async_client=async_client)

    def get_model_info(self) -> dict[str, Any]:
        """Return model info with the resolved provider name."""
        info = super().get_model_info()
        info["provider"] = self.provider_name.capitalize()
        return info

    def __repr__(self) -> str:
        return (
            f"OpenAICompatLLM(provider='{self.provider_name}', "
            f"model='{self.config.model}', "
            f"temperature={self.config.temperature})"
        )


class DeepSeekLLM(OpenAICompatLLM):
    """DeepSeek provider (deepseek-chat, deepseek-reasoner)."""

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        super().__init__(config=config, provider="deepseek", **kwargs)


class KimiLLM(OpenAICompatLLM):
    """Kimi / Moonshot provider (moonshot-v1-*, kimi-latest)."""

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        super().__init__(config=config, provider="kimi", **kwargs)


class QwenLLM(OpenAICompatLLM):
    """Qwen / DashScope provider (qwen-max, qwen-plus, qwen-turbo)."""

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        super().__init__(config=config, provider="qwen", **kwargs)


class ZhipuLLM(OpenAICompatLLM):
    """Zhipu / GLM provider (glm-4-plus, glm-4-flash)."""

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        super().__init__(config=config, provider="zhipu", **kwargs)


class LMStudioLLM(OpenAICompatLLM):
    """LM Studio local provider."""

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        super().__init__(config=config, provider="lmstudio", **kwargs)


class CustomLLM(OpenAICompatLLM):
    """Custom OpenAI-compatible endpoint provider."""

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        if config and not config.base_url and "base_url" not in kwargs:
            raise ValueError("CustomLLM requires a base_url. Pass it via LLMConfig or as a kwarg.")
        super().__init__(config=config, provider="custom", **kwargs)


__all__ = [
    "CustomLLM",
    "DeepSeekLLM",
    "KimiLLM",
    "LMStudioLLM",
    "OpenAICompatLLM",
    "QwenLLM",
    "ZhipuLLM",
]
