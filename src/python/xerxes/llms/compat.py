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
    """Open aicompat llm.

    Inherits from: OpenAILLM
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        provider: str | None = None,
        client: Any | None = None,
        async_client: Any | None = None,
        **kwargs,
    ):
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            config (LLMConfig | None, optional): IN: config. Defaults to None. OUT: Consumed during execution.
            provider (str | None, optional): IN: provider. Defaults to None. OUT: Consumed during execution.
            client (Any | None, optional): IN: client. Defaults to None. OUT: Consumed during execution.
            async_client (Any | None, optional): IN: async client. Defaults to None. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            Any: OUT: Result of the operation."""

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
        """Retrieve the model info.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

        info = super().get_model_info()
        info["provider"] = self.provider_name.capitalize()
        return info

    def __repr__(self) -> str:
        """Dunder method for repr.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            str: OUT: Result of the operation."""
        return (
            f"OpenAICompatLLM(provider='{self.provider_name}', "
            f"model='{self.config.model}', "
            f"temperature={self.config.temperature})"
        )


class DeepSeekLLM(OpenAICompatLLM):
    """Deep seek llm.

    Inherits from: OpenAICompatLLM
    """

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            config (LLMConfig | None, optional): IN: config. Defaults to None. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            Any: OUT: Result of the operation."""
        super().__init__(config=config, provider="deepseek", **kwargs)


class MiniMaxLLM(OpenAICompatLLM):
    """Mini max llm.

    Inherits from: OpenAICompatLLM
    """

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            config (LLMConfig | None, optional): IN: config. Defaults to None. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            Any: OUT: Result of the operation."""
        super().__init__(config=config, provider="minimax", **kwargs)


class KimiLLM(OpenAICompatLLM):
    """Kimi llm.

    Inherits from: OpenAICompatLLM
    """

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            config (LLMConfig | None, optional): IN: config. Defaults to None. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            Any: OUT: Result of the operation."""
        super().__init__(config=config, provider="kimi", **kwargs)


class QwenLLM(OpenAICompatLLM):
    """Qwen llm.

    Inherits from: OpenAICompatLLM
    """

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            config (LLMConfig | None, optional): IN: config. Defaults to None. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            Any: OUT: Result of the operation."""
        super().__init__(config=config, provider="qwen", **kwargs)


class ZhipuLLM(OpenAICompatLLM):
    """Zhipu llm.

    Inherits from: OpenAICompatLLM
    """

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            config (LLMConfig | None, optional): IN: config. Defaults to None. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            Any: OUT: Result of the operation."""
        super().__init__(config=config, provider="zhipu", **kwargs)


class LMStudioLLM(OpenAICompatLLM):
    """Lmstudio llm.

    Inherits from: OpenAICompatLLM
    """

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            config (LLMConfig | None, optional): IN: config. Defaults to None. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            Any: OUT: Result of the operation."""
        super().__init__(config=config, provider="lmstudio", **kwargs)


class CustomLLM(OpenAICompatLLM):
    """Custom llm.

    Inherits from: OpenAICompatLLM
    """

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            config (LLMConfig | None, optional): IN: config. Defaults to None. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            Any: OUT: Result of the operation."""
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
