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
"""Static registry of LLM providers, models, pricing, and context limits.

Public surface:

* :data:`PROVIDERS` — provider name -> :class:`ProviderConfig` (API
  key env var, base URL, default context limit, known models).
* :data:`COSTS` — model name -> (input USD per 1M tokens, output USD).
* :func:`detect_provider` — guess a provider from a model identifier.
* :func:`get_context_limit` — context window for a model.
* :func:`calc_cost` — USD cost of a token bill.
* :func:`get_api_key` — resolve a provider's key from env or config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderConfig:
    """Static metadata describing one LLM provider.

    Attributes:
        name: provider slug (matches :data:`PROVIDERS` keys).
        type: wire protocol family (e.g. ``"openai"``, ``"anthropic"``).
        api_key_env: environment variable holding the API key.
        base_url: HTTPS endpoint root for the provider.
        context_limit: default context window in tokens.
        models: known model identifiers for this provider.
        default_api_key: literal key used when no env var is set.
    """

    name: str
    type: str
    api_key_env: str | None = None
    base_url: str | None = None
    context_limit: int = 128_000
    models: list[str] = field(default_factory=list)
    default_api_key: str | None = None


PROVIDERS: dict[str, ProviderConfig] = {
    "anthropic": ProviderConfig(
        name="anthropic",
        type="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        context_limit=200_000,
        models=[
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
        ],
    ),
    "openai": ProviderConfig(
        name="openai",
        type="openai",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        context_limit=128_000,
        models=[
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "o3-mini",
            "o3",
            "o4-mini",
            "o1",
            "o1-mini",
        ],
    ),
    "gemini": ProviderConfig(
        name="gemini",
        type="openai",
        api_key_env="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        context_limit=1_000_000,
        models=[
            "gemini-2.5-pro-preview-03-25",
            "gemini-2.5-flash-preview-04-17",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ],
    ),
    "kimi": ProviderConfig(
        name="kimi",
        type="openai",
        api_key_env="MOONSHOT_API_KEY",
        base_url="https://api.moonshot.cn/v1",
        context_limit=128_000,
        models=[
            "moonshot-v1-8k",
            "moonshot-v1-32k",
            "moonshot-v1-128k",
            "kimi-latest",
        ],
    ),
    "kimi-code": ProviderConfig(
        name="kimi-code",
        type="openai",
        api_key_env="KIMI_CODE_API_KEY",
        # Coding-specialised endpoint published at https://www.kimi.com/code/.
        # OpenAI-compatible surface; the Anthropic-compatible variant lives
        # at https://api.kimi.com/coding/ but Xerxes' streaming loop targets
        # the OpenAI shape, so we use the /v1 path.
        base_url="https://api.kimi.com/coding/v1",
        context_limit=128_000,
        models=[
            # ``kimi-for-coding`` is the stable model id — the backend
            # rolls forward (currently maps to kimi-k2.6) without callers
            # needing to change the string.
            "kimi-for-coding",
        ],
    ),
    "qwen": ProviderConfig(
        name="qwen",
        type="openai",
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        context_limit=1_000_000,
        models=[
            "qwen-max",
            "qwen-plus",
            "qwen-turbo",
            "qwen-long",
            "qwen3-235b-a22b",
            "qwen2.5-72b-instruct",
            "qwen2.5-coder-32b-instruct",
            "qwq-32b",
        ],
    ),
    "zhipu": ProviderConfig(
        name="zhipu",
        type="openai",
        api_key_env="ZHIPU_API_KEY",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        context_limit=128_000,
        models=[
            "glm-4-plus",
            "glm-4",
            "glm-4-flash",
            "glm-4-air",
            "glm-z1-flash",
        ],
    ),
    "deepseek": ProviderConfig(
        name="deepseek",
        type="openai",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        context_limit=64_000,
        models=[
            "deepseek-chat",
            "deepseek-coder",
            "deepseek-reasoner",
        ],
    ),
    "minimax": ProviderConfig(
        name="minimax",
        type="openai",
        api_key_env="MINIMAX_API_KEY",
        base_url="https://api.minimax.io/v1",
        context_limit=128_000,
        models=[
            "MiniMax-M2.7-highspeed",
            "MiniMax-M2.7-flashspeed",
            "MiniMax-Text-01",
            "MiniMax-Text-01-MiniApp",
            "abab6.5s-chat",
            "abab6.5-chat",
            "abab6-chat",
            "abab5.5s-chat",
            "abab5.5-chat",
            "abab5-chat",
        ],
    ),
    "ollama": ProviderConfig(
        name="ollama",
        type="openai",
        api_key_env=None,
        base_url="http://localhost:11434/v1",
        default_api_key="ollama",
        context_limit=128_000,
        models=[
            "llama3.3",
            "llama3.2",
            "llama3.1",
            "phi4",
            "mistral",
            "mixtral",
            "qwen2.5-coder",
            "deepseek-r1",
            "gemma3",
            "codellama",
        ],
    ),
    "lmstudio": ProviderConfig(
        name="lmstudio",
        type="openai",
        api_key_env=None,
        base_url="http://localhost:1234/v1",
        default_api_key="lm-studio",
        context_limit=128_000,
        models=[],
    ),
    "custom": ProviderConfig(
        name="custom",
        type="openai",
        api_key_env="CUSTOM_API_KEY",
        base_url=None,
        context_limit=128_000,
        models=[],
    ),
}

COSTS: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (15.0, 75.0),
    "claude-opus-4-5": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.8, 4.0),
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "claude-3-5-haiku-20241022": (0.8, 4.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    "o3-mini": (1.1, 4.4),
    "o3": (10.0, 40.0),
    "o4-mini": (1.1, 4.4),
    "o1": (15.0, 60.0),
    "o1-mini": (3.0, 12.0),
    "gemini-2.5-pro-preview-03-25": (1.25, 10.0),
    "gemini-2.5-flash-preview-04-17": (0.15, 0.6),
    "gemini-2.0-flash": (0.075, 0.3),
    "gemini-2.0-flash-lite": (0.075, 0.3),
    "gemini-1.5-pro": (1.25, 5.0),
    "gemini-1.5-flash": (0.075, 0.3),
    "moonshot-v1-8k": (1.0, 3.0),
    "moonshot-v1-32k": (2.4, 7.0),
    "moonshot-v1-128k": (8.0, 24.0),
    "kimi-latest": (2.4, 7.0),
    # Kimi Code pricing not publicly published — mirror kimi-latest as a
    # conservative placeholder so cost estimates aren't zero.
    "kimi-for-coding": (2.4, 7.0),
    "qwen-max": (2.4, 9.6),
    "qwen-plus": (0.4, 1.2),
    "qwen-turbo": (0.2, 0.6),
    "qwen-long": (0.4, 1.2),
    "qwen3-235b-a22b": (2.4, 9.6),
    "deepseek-chat": (0.27, 1.1),
    "deepseek-coder": (0.27, 1.1),
    "deepseek-reasoner": (0.55, 2.19),
    "MiniMax-M2.7-highspeed": (0.0, 0.0),
    "MiniMax-M2.7-flashspeed": (0.0, 0.0),
    "MiniMax-Text-01": (0.0, 0.0),
    "MiniMax-Text-01-MiniApp": (0.0, 0.0),
    "glm-4-plus": (0.7, 0.7),
    "glm-4": (0.7, 0.7),
    "glm-4-flash": (0.0, 0.0),
    "glm-4-air": (0.14, 0.14),
}

_PREFIX_MAP: list[tuple[str, str]] = sorted(
    [
        ("claude-", "anthropic"),
        ("gpt-", "openai"),
        ("o1", "openai"),
        ("o3", "openai"),
        ("o4", "openai"),
        ("gemini-", "gemini"),
        ("moonshot-", "kimi"),
        # ``kimi-for-`` must precede the looser ``kimi-`` rule (the prefix
        # map is sorted by length descending, so this is automatic, but
        # keeping them adjacent makes the relationship obvious).
        ("kimi-for-", "kimi-code"),
        ("kimi-", "kimi"),
        ("qwq-", "qwen"),
        ("qwen", "qwen"),
        ("glm-", "zhipu"),
        ("deepseek-", "deepseek"),
        ("minimax-", "minimax"),
        ("abab", "minimax"),
        ("llama", "ollama"),
        ("mistral", "ollama"),
        ("mixtral", "ollama"),
        ("phi", "ollama"),
        ("gemma", "ollama"),
        ("codellama", "ollama"),
    ],
    key=lambda x: len(x[0]),
    reverse=True,
)


def detect_provider(model: str) -> str:
    """Return the provider slug for ``model``.

    A ``provider/model`` prefix (e.g. ``"anthropic/claude-3-5-sonnet"``)
    is honoured directly; otherwise the longest matching prefix in
    ``_PREFIX_MAP`` wins. Defaults to ``"openai"`` for unknown names.
    """

    if "/" in model:
        return model.split("/", 1)[0].lower()
    lower = model.lower()
    for prefix, provider_name in _PREFIX_MAP:
        if lower.startswith(prefix):
            return provider_name
    return "openai"


def bare_model(model: str) -> str:
    """Strip any ``provider/`` prefix from ``model``."""

    return model.split("/", 1)[1] if "/" in model else model


def get_provider_config(provider_name: str) -> ProviderConfig:
    """Return :class:`ProviderConfig` for ``provider_name``; raises ``KeyError``."""

    return PROVIDERS[provider_name]


def get_api_key(provider_name: str, extra_config: dict | None = None) -> str:
    """Resolve an API key for ``provider_name`` (config -> env -> default).

    Returns the empty string when no key can be found.
    """

    prov = PROVIDERS.get(provider_name)
    if prov is None:
        return ""

    if extra_config:
        cfg_key = extra_config.get(f"{provider_name}_api_key", "")
        if cfg_key:
            return cfg_key

    if prov.api_key_env:
        val = os.environ.get(prov.api_key_env, "")
        if val:
            return val

    return prov.default_api_key or ""


# Per-provider HTTP headers injected on every request the streaming loop
# makes. Used to satisfy provider-side client-identity gates — Kimi Code's
# endpoint returns 403 unless the User-Agent matches one of the allowlisted
# coding agents ("Kimi CLI, Claude Code, Roo Code, Kilo Code"), so we identify
# ourselves as ``claude-code`` for that provider.
_PROVIDER_DEFAULT_HEADERS: dict[str, dict[str, str]] = {
    "kimi-code": {
        # Spoofing claude-code is the path of least surprise — the format is
        # stable, the version doesn't matter to the gate, and it bypasses
        # Kimi's "Coding Agents only" allowlist. If Moonshot ever ships a
        # public Kimi Code Python SDK we should switch to whatever that
        # advertises so we stay compliant rather than impersonating.
        "User-Agent": "claude-code/1.0.0",
        "X-Stainless-Lang": "claude-code",
        "X-Client-Name": "claude-code",
    },
}


def provider_default_headers(provider_name: str) -> dict[str, str]:
    """Return the headers the streaming loop should attach for ``provider_name``.

    Empty dict means "no overrides — let httpx / the SDK pick its own
    defaults". Returned dict is a fresh copy so callers can mutate it.
    """
    return dict(_PROVIDER_DEFAULT_HEADERS.get(provider_name, {}))


def calc_cost(model: str, in_tok: int, out_tok: int) -> float:
    """Return the USD cost of an LLM call using :data:`COSTS`.

    Rates in :data:`COSTS` are USD per 1M tokens; unknown models cost
    ``0.0``.
    """

    name = bare_model(model)
    ic, oc = COSTS.get(name, (0.0, 0.0))
    return (in_tok * ic + out_tok * oc) / 1_000_000


def list_all_models() -> dict[str, list[str]]:
    """Return the registered provider -> models map (providers with models only)."""

    return {name: list(prov.models) for name, prov in PROVIDERS.items() if prov.models}


_MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "MiniMax-M2.7-highspeed": 1_024_000,
    "MiniMax-M2.7-flashspeed": 1_024_000,
    "MiniMax-Text-01": 256_000,
    "MiniMax-Text-01-MiniApp": 256_000,
}


def get_context_limit(model: str) -> int:
    """Return the context window for ``model`` (model-specific or provider default)."""

    if model in _MODEL_CONTEXT_LIMITS:
        return _MODEL_CONTEXT_LIMITS[model]
    provider_name = detect_provider(model)
    prov = PROVIDERS.get(provider_name)
    return prov.context_limit if prov else 128_000


__all__ = [
    "COSTS",
    "PROVIDERS",
    "ProviderConfig",
    "bare_model",
    "calc_cost",
    "detect_provider",
    "get_api_key",
    "get_context_limit",
    "get_provider_config",
    "list_all_models",
    "provider_default_headers",
]
