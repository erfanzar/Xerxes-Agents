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
"""LLM provider registry tests."""

from xerxes.llms.registry import PROVIDERS, get_context_limit, provider_model, resolve_provider


def test_resolve_provider_promotes_kimi_for_coding_model() -> None:
    assert resolve_provider("kimi/kimi-for-coding") == "kimi-code"


def test_resolve_provider_promotes_kimi_coding_endpoint() -> None:
    assert resolve_provider("kimi/kimi-latest", {"base_url": "https://api.kimi.com/coding/v1"}) == "kimi-code"


def test_resolve_provider_promotes_openrouter_endpoint() -> None:
    assert (
        resolve_provider(
            "anthropic/claude-sonnet-4.5",
            {"base_url": "https://openrouter.ai/api/v1"},
        )
        == "openrouter"
    )


def test_openrouter_provider_config() -> None:
    cfg = PROVIDERS["openrouter"]
    assert cfg.type == "openai"
    assert cfg.api_key_env == "OPENROUTER_API_KEY"
    assert cfg.base_url == "https://openrouter.ai/api/v1"


def test_provider_model_preserves_openrouter_namespaced_model_ids() -> None:
    assert provider_model("anthropic/claude-sonnet-4.5", "openrouter") == "anthropic/claude-sonnet-4.5"
    assert provider_model("openrouter/anthropic/claude-sonnet-4.5", "openrouter") == "anthropic/claude-sonnet-4.5"
    assert provider_model("openrouter/auto", "openrouter") == "openrouter/auto"


def test_resolve_provider_keeps_generic_kimi_endpoint() -> None:
    assert resolve_provider("kimi/kimi-latest", {"base_url": "https://api.moonshot.cn/v1"}) == "kimi"


def test_context_limit_uses_effective_provider_for_kimi_code_prefix() -> None:
    assert get_context_limit("kimi/kimi-for-coding") == 256_000
