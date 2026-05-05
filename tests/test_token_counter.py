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
"""Tests for xerxes.context.token_counter module."""

from xerxes.context.token_counter import (
    ProviderTokenCounter,
    SmartTokenCounter,
)


class TestProviderTokenCounter:
    def test_detect_provider_openai(self):
        assert ProviderTokenCounter._detect_provider("gpt-4") == "openai"
        assert ProviderTokenCounter._detect_provider("gpt-3.5-turbo") == "openai"
        assert ProviderTokenCounter._detect_provider("o1-preview") == "openai"

    def test_detect_provider_anthropic(self):
        assert ProviderTokenCounter._detect_provider("claude-3-sonnet") == "anthropic"

    def test_detect_provider_google(self):
        assert ProviderTokenCounter._detect_provider("gemini-pro") == "google"
        assert ProviderTokenCounter._detect_provider("palm-2") == "google"

    def test_detect_provider_meta(self):
        assert ProviderTokenCounter._detect_provider("llama-2-70b") == "meta"

    def test_detect_provider_mistral(self):
        assert ProviderTokenCounter._detect_provider("mistral-7b") == "mistral"
        assert ProviderTokenCounter._detect_provider("mixtral-8x7b") == "mistral"

    def test_detect_provider_unknown(self):
        assert ProviderTokenCounter._detect_provider("unknown-model") is None

    def test_detect_provider_empty(self):
        assert ProviderTokenCounter._detect_provider("") is None

    def test_messages_to_text(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = ProviderTokenCounter._messages_to_text(messages)
        assert "user: Hello" in result
        assert "assistant: Hi" in result

    def test_messages_to_text_missing_fields(self):
        messages = [{"content": "text"}, {"role": "user"}]
        result = ProviderTokenCounter._messages_to_text(messages)
        assert ": text" in result
        assert "user: " in result

    def test_count_fallback_tokens(self):
        text = "a" * 100
        result = ProviderTokenCounter._count_fallback_tokens(text)
        assert result > 0

    def test_count_tokens_for_provider_with_messages(self):
        messages = [{"role": "user", "content": "Hello world"}]
        result = ProviderTokenCounter.count_tokens_for_provider(messages, provider="openai", model="gpt-4")
        assert result > 0

    def test_count_tokens_for_provider_detect_from_model(self):
        result = ProviderTokenCounter.count_tokens_for_provider("Hello", model="gpt-4")
        assert result > 0

    def test_count_tokens_for_provider_no_provider(self):
        result = ProviderTokenCounter.count_tokens_for_provider("Hello", provider=None, model=None)
        assert result > 0

    def test_count_openai_tokens_gpt4o(self):
        result = ProviderTokenCounter._count_openai_tokens("Hello world", model="gpt-4o")
        assert result > 0

    def test_count_openai_tokens_gpt4(self):
        result = ProviderTokenCounter._count_openai_tokens("Hello world", model="gpt-4")
        assert result > 0

    def test_count_openai_tokens_gpt35(self):
        result = ProviderTokenCounter._count_openai_tokens("Hello world", model="gpt-3.5-turbo")
        assert result > 0

    def test_count_openai_tokens_no_model(self):
        result = ProviderTokenCounter._count_openai_tokens("Hello world")
        assert result > 0

    def test_count_anthropic_tokens_fallback(self):
        result = ProviderTokenCounter._count_anthropic_tokens("Hello world", model="claude-3")
        assert result > 0

    def test_count_google_tokens(self):
        result = ProviderTokenCounter._count_google_tokens("Hello world", model="gemini-pro")
        assert result > 0


class TestSmartTokenCounter:
    def test_init_auto_detect(self):
        counter = SmartTokenCounter(model="gpt-4")
        assert counter.provider == "openai"

    def test_init_explicit_provider(self):
        counter = SmartTokenCounter(provider="anthropic", model="claude-3")
        assert counter.provider == "anthropic"

    def test_init_no_model(self):
        counter = SmartTokenCounter()
        assert counter.provider is None

    def test_count_tokens(self):
        counter = SmartTokenCounter(model="gpt-4")
        result = counter.count_tokens("Hello world")
        assert result > 0

    def test_count_tokens_messages(self):
        counter = SmartTokenCounter(model="gpt-4")
        messages = [{"role": "user", "content": "Hello"}]
        result = counter.count_tokens(messages)
        assert result > 0

    def test_count_remaining_capacity(self):
        counter = SmartTokenCounter(model="gpt-4")
        remaining = counter.count_remaining_capacity("Hello", max_tokens=4096)
        assert remaining > 0
        assert remaining < 4096

    def test_count_remaining_capacity_exceeded(self):
        counter = SmartTokenCounter(model="gpt-4")
        remaining = counter.count_remaining_capacity("Hello " * 1000, max_tokens=1)
        assert remaining == 0

    def test_estimate_compression_ratio(self):
        counter = SmartTokenCounter(model="gpt-4")
        ratio = counter.estimate_compression_ratio("Hello world this is a test", "Hello test")
        assert 0.0 < ratio < 1.0

    def test_estimate_compression_ratio_empty_original(self):
        counter = SmartTokenCounter(model="gpt-4")
        ratio = counter.estimate_compression_ratio("", "something")
        assert ratio == 0.0

    def test_estimate_compression_ratio_same_text(self):
        counter = SmartTokenCounter(model="gpt-4")
        ratio = counter.estimate_compression_ratio("Hello", "Hello")
        assert ratio == 0.0
