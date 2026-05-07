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
"""Compat module for Xerxes.

Exports:
    - OpenAICompatLLM
    - DeepSeekLLM
    - MiniMaxLLM
    - KimiLLM
    - QwenLLM
    - ZhipuLLM
    - LMStudioLLM
    - CustomLLM"""

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
    """OpenAI-compatible LLM client for third-party providers.

    Subclass of OpenAILLM that auto-detects provider settings (base URL,
    API key environment variable, context limits) from the registry.

    Attributes:
        provider_name: The detected or specified provider name.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        provider: str | None = None,
        client: Any | None = None,
        async_client: Any | None = None,
        **kwargs,
    ):
        """Initialize the OpenAI-compatible LLM client.

        Args:
            config: Optional LLM configuration.
            provider: Provider name for registry lookup. If None, inferred from model name.
            client: Optional pre-configured sync OpenAI client.
            async_client: Optional pre-configured async OpenAI client.
            **kwargs: Additional configuration fields forwarded to LLMConfig.
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
        """Return model information with the provider name capitalized."""
        info = super().get_model_info()
        info["provider"] = self.provider_name.capitalize()
        return info

    def __repr__(self) -> str:
        """Return a human-readable representation of the client."""
        return (
            f"OpenAICompatLLM(provider='{self.provider_name}', "
            f"model='{self.config.model}', "
            f"temperature={self.config.temperature})"
        )


class DeepSeekLLM(OpenAICompatLLM):
    """DeepSeek LLM client using the OpenAI-compatible API.

    Inherits from OpenAICompatLLM with provider="deepseek".
    """

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize the DeepSeek LLM client.

        Args:
            config: Optional LLM configuration.
            **kwargs: Additional configuration fields forwarded to the parent class.
        """
        super().__init__(config=config, provider="deepseek", **kwargs)


class MiniMaxLLM(OpenAICompatLLM):
    """MiniMax LLM client using the OpenAI-compatible API.

    Inherits from OpenAICompatLLM with provider="minimax".
    """

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize the MiniMax LLM client.

        Args:
            config: Optional LLM configuration.
            **kwargs: Additional configuration fields forwarded to the parent class.
        """
        super().__init__(config=config, provider="minimax", **kwargs)


class KimiLLM(OpenAICompatLLM):
    """Kimi (Moonshot) LLM client using the OpenAI-compatible API.

    Inherits from OpenAICompatLLM with provider="kimi".
    """

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize the Kimi LLM client.

        Args:
            config: Optional LLM configuration.
            **kwargs: Additional configuration fields forwarded to the parent class.
        """
        super().__init__(config=config, provider="kimi", **kwargs)


class QwenLLM(OpenAICompatLLM):
    """Qwen (Alibaba DashScope) LLM client using the OpenAI-compatible API.

    Inherits from OpenAICompatLLM with provider="qwen".
    """

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize the Qwen LLM client.

        Args:
            config: Optional LLM configuration.
            **kwargs: Additional configuration fields forwarded to the parent class.
        """
        super().__init__(config=config, provider="qwen", **kwargs)


class ZhipuLLM(OpenAICompatLLM):
    """Zhipu (GLM) LLM client using the OpenAI-compatible API.

    Inherits from OpenAICompatLLM with provider="zhipu".
    """

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize the Zhipu LLM client.

        Args:
            config: Optional LLM configuration.
            **kwargs: Additional configuration fields forwarded to the parent class.
        """
        super().__init__(config=config, provider="zhipu", **kwargs)


class LMStudioLLM(OpenAICompatLLM):
    """LM Studio LLM client using the OpenAI-compatible local API.

    Inherits from OpenAICompatLLM with provider="lmstudio".
    """

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize the LM Studio LLM client.

        Args:
            config: Optional LLM configuration.
            **kwargs: Additional configuration fields forwarded to the parent class.
        """
        super().__init__(config=config, provider="lmstudio", **kwargs)


class CustomLLM(OpenAICompatLLM):
    """Custom LLM client for user-defined OpenAI-compatible endpoints.

    Requires a base_url to be provided. Inherits from OpenAICompatLLM
    with provider="custom".
    """

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize the Custom LLM client.

        Args:
            config: Optional LLM configuration.
            **kwargs: Additional configuration fields forwarded to the parent class.

        Raises:
            ValueError: If no base_url is provided via config or kwargs.
        """
        if config and not config.base_url and "base_url" not in kwargs:
            raise ValueError("CustomLLM requires a base_url. Pass it via LLMConfig or as a kwarg.")
        super().__init__(config=config, provider="custom", **kwargs)


__all__ = [
    "CustomLLM",
    "DeepSeekLLM",
    "KimiLLM",
    "LMStudioLLM",
    "MiniMaxLLM",
    "OpenAICompatLLM",
    "QwenLLM",
    "ZhipuLLM",
]
