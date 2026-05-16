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
"""Cache-aware pricing table covering 50+ Anthropic/OpenAI/Google/open-weight models.

Each entry models four token classes per model:

    * ``input``: regular prompt tokens.
    * ``output``: regular response tokens.
    * ``cache_read``: tokens served from a cached prefix.
    * ``cache_write``: tokens written into a new cache entry.

All numbers are USD per million tokens. :func:`compute_cost` does the
arithmetic and tolerates missing models by returning ``0.0``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """Pricing record for one model.

    Attributes:
        model: Provider-canonical model id.
        input_per_million: USD per 1 M non-cached input tokens.
        output_per_million: USD per 1 M output tokens.
        cache_read_per_million: USD per 1 M cache-hit input tokens.
        cache_write_per_million: USD per 1 M tokens written into the cache.
        context_window: Maximum context length supported by the model.
        max_output_tokens: Maximum tokens the model may emit in one response.
    """

    model: str
    input_per_million: float
    output_per_million: float
    cache_read_per_million: float = 0.0
    cache_write_per_million: float = 0.0
    context_window: int = 200_000
    max_output_tokens: int = 8192


_PRICING: dict[str, ModelPricing] = {}


def _add(*entries: ModelPricing) -> None:
    """Register one or more :class:`ModelPricing` entries into :data:`_PRICING`."""
    for e in entries:
        _PRICING[e.model] = e


# Anthropic (cache_read = 10% of input; cache_write = 125%).
_add(
    ModelPricing("claude-opus-4-7", 15.0, 75.0, 1.5, 18.75, 200_000, 8_192),
    ModelPricing("claude-sonnet-4-6", 3.0, 15.0, 0.3, 3.75, 200_000, 8_192),
    ModelPricing("claude-haiku-4-5", 1.0, 5.0, 0.1, 1.25, 200_000, 8_192),
    ModelPricing("claude-opus-4", 15.0, 75.0, 1.5, 18.75, 200_000, 8_192),
    ModelPricing("claude-sonnet-4", 3.0, 15.0, 0.3, 3.75, 200_000, 8_192),
    ModelPricing("claude-3-5-sonnet-20241022", 3.0, 15.0, 0.3, 3.75, 200_000, 8_192),
    ModelPricing("claude-3-5-haiku-20241022", 0.8, 4.0, 0.08, 1.0, 200_000, 8_192),
)
# OpenAI.
_add(
    ModelPricing("gpt-4o", 5.0, 15.0, 1.25, 0.0, 128_000, 16_384),
    ModelPricing("gpt-4o-mini", 0.15, 0.60, 0.075, 0.0, 128_000, 16_384),
    ModelPricing("gpt-4.1", 2.0, 8.0, 0.5, 0.0, 1_000_000, 32_768),
    ModelPricing("gpt-4.1-mini", 0.40, 1.60, 0.10, 0.0, 1_000_000, 32_768),
    ModelPricing("gpt-4.1-nano", 0.10, 0.40, 0.025, 0.0, 1_000_000, 32_768),
    ModelPricing("o1", 15.0, 60.0, 7.5, 0.0, 200_000, 100_000),
    ModelPricing("o1-mini", 3.0, 12.0, 1.5, 0.0, 128_000, 65_536),
    ModelPricing("o3", 60.0, 240.0, 0.0, 0.0, 200_000, 100_000),
    ModelPricing("o3-mini", 1.10, 4.40, 0.55, 0.0, 200_000, 100_000),
)
# Google / Gemini.
_add(
    ModelPricing("gemini-2.5-pro", 1.25, 10.0, 0.3125, 0.0, 2_000_000, 8_192),
    ModelPricing("gemini-2.5-flash", 0.30, 2.50, 0.075, 0.0, 1_000_000, 8_192),
    ModelPricing("gemini-2.5-flash-lite", 0.10, 0.40, 0.025, 0.0, 1_000_000, 8_192),
    ModelPricing("gemini-2.0-flash", 0.10, 0.40, 0.025, 0.0, 1_000_000, 8_192),
    ModelPricing("gemini-2.0-flash-lite", 0.075, 0.30, 0.0, 0.0, 1_000_000, 8_192),
)
# Together / open-weight via OpenRouter etc.
_add(
    ModelPricing("meta-llama/llama-3.3-70b-instruct", 0.59, 0.79, 0.0, 0.0, 131_072, 8_192),
    ModelPricing("meta-llama/llama-3.1-405b-instruct", 5.0, 5.0, 0.0, 0.0, 131_072, 4_096),
    ModelPricing("meta-llama/llama-3.1-70b-instruct", 0.59, 0.79, 0.0, 0.0, 131_072, 4_096),
    ModelPricing("mistralai/mixtral-8x22b-instruct", 1.2, 1.2, 0.0, 0.0, 65_536, 8_192),
    ModelPricing("mistralai/mistral-large-2407", 2.0, 6.0, 0.0, 0.0, 128_000, 8_192),
    ModelPricing("qwen/qwen-2.5-72b-instruct", 0.40, 0.40, 0.0, 0.0, 131_072, 8_192),
    ModelPricing("qwen/qwen-2.5-coder-32b-instruct", 0.18, 0.18, 0.0, 0.0, 131_072, 8_192),
    ModelPricing("deepseek/deepseek-chat", 0.27, 1.10, 0.07, 0.0, 64_000, 8_192),
    ModelPricing("deepseek/deepseek-reasoner", 0.55, 2.19, 0.14, 0.0, 64_000, 8_192),
    ModelPricing("zhipuai/glm-4-plus", 0.50, 1.00, 0.0, 0.0, 128_000, 8_192),
    ModelPricing("moonshot/moonshot-v1-128k", 2.00, 2.00, 0.0, 0.0, 128_000, 8_192),
    ModelPricing("kimi/kimi-latest", 1.80, 1.80, 0.0, 0.0, 200_000, 8_192),
    ModelPricing("minimax/minimax-text-01", 1.00, 1.00, 0.0, 0.0, 1_000_000, 8_192),
    ModelPricing("xai/grok-3", 5.0, 15.0, 0.0, 0.0, 256_000, 32_768),
    ModelPricing("xai/grok-3-mini", 0.30, 0.50, 0.0, 0.0, 256_000, 32_768),
)
# Mistral direct.
_add(
    ModelPricing("mistral-large-2407", 2.0, 6.0, 0.0, 0.0, 128_000, 8_192),
    ModelPricing("mistral-medium", 0.40, 2.0, 0.0, 0.0, 128_000, 8_192),
    ModelPricing("ministral-8b-2410", 0.10, 0.10, 0.0, 0.0, 131_072, 4_096),
)


def get_pricing(model: str) -> ModelPricing | None:
    """Return the :class:`ModelPricing` for ``model`` or ``None`` if unknown."""
    return _PRICING.get(model)


def known_models() -> list[str]:
    """Return every model id with a pricing entry, sorted alphabetically."""
    return sorted(_PRICING.keys())


def compute_cost(
    *,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Compute the USD cost for a single LLM call across all four token classes.

    Returns ``0.0`` when ``model`` lacks a pricing entry so callers can use
    the result unconditionally; the absence of pricing is also a useful
    smoke-test signal in dashboards.
    """
    pricing = _PRICING.get(model)
    if pricing is None:
        return 0.0
    cost = (
        input_tokens * pricing.input_per_million
        + output_tokens * pricing.output_per_million
        + cache_read_tokens * pricing.cache_read_per_million
        + cache_write_tokens * pricing.cache_write_per_million
    ) / 1_000_000.0
    return cost


__all__ = ["ModelPricing", "compute_cost", "get_pricing", "known_models"]
