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

from __future__ import annotations

from types import SimpleNamespace

import pytest
from xerxes.streaming import loop


def test_explicit_base_url_uses_direct_http_client_and_no_retries(monkeypatch) -> None:
    import openai

    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            raise RuntimeError("stop before network")

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    with pytest.raises(RuntimeError, match="stop before network"):
        list(
            loop._stream_openai_compat(
                model="qwen",
                system="",
                messages=[{"role": "user", "content": "hi"}],
                tool_schemas=[],
                config={"base_url": "http://localhost:8000/v1"},
                provider_name="openai",
            )
        )

    assert captured["base_url"] == "http://localhost:8000/v1"
    assert captured["max_retries"] == 0
    assert captured["http_client"]._trust_env is False
    captured["http_client"].close()


def test_builtin_provider_keeps_sdk_retry_default(monkeypatch) -> None:
    import openai

    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            raise RuntimeError("stop before network")

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    with pytest.raises(RuntimeError, match="stop before network"):
        list(
            loop._stream_openai_compat(
                model="gpt-4o-mini",
                system="",
                messages=[{"role": "user", "content": "hi"}],
                tool_schemas=[],
                config={},
                provider_name="openai",
            )
        )

    assert captured["max_retries"] == 2
    assert "http_client" not in captured


def test_stream_llm_resolves_kimi_code_saved_profile(monkeypatch) -> None:
    captured: dict = {}

    def fake_stream_openai_compat(
        model: str,
        system: str,
        messages: list[dict],
        tool_schemas: list[dict],
        config: dict,
        provider_name: str,
    ):
        captured.update(
            {
                "model": model,
                "provider_name": provider_name,
                "base_url": config["base_url"],
            }
        )
        yield {"tool_calls": [], "input_tokens": 1, "output_tokens": 1}

    monkeypatch.setattr(loop, "_stream_openai_compat", fake_stream_openai_compat)

    result = list(
        loop._stream_llm(
            model="kimi/kimi-for-coding",
            provider_type="openai",
            system="",
            messages=[{"role": "user", "content": "hi"}],
            tool_schemas=[],
            config={"base_url": "https://api.kimi.com/coding/v1"},
        )
    )

    assert result[-1]["tool_calls"] == []
    assert captured == {
        "model": "kimi-for-coding",
        "provider_name": "kimi-code",
        "base_url": "https://api.kimi.com/coding/v1",
    }


def test_stream_llm_preserves_openrouter_namespaced_model(monkeypatch) -> None:
    captured: dict = {}

    def fake_stream_openai_compat(
        model: str,
        system: str,
        messages: list[dict],
        tool_schemas: list[dict],
        config: dict,
        provider_name: str,
    ):
        captured.update(
            {
                "model": model,
                "provider_name": provider_name,
                "base_url": config["base_url"],
            }
        )
        yield {"tool_calls": [], "input_tokens": 1, "output_tokens": 1}

    monkeypatch.setattr(loop, "_stream_openai_compat", fake_stream_openai_compat)

    result = list(
        loop._stream_llm(
            model="anthropic/claude-sonnet-4.5",
            provider_type="openai",
            system="",
            messages=[{"role": "user", "content": "hi"}],
            tool_schemas=[],
            config={"base_url": "https://openrouter.ai/api/v1"},
        )
    )

    assert result[-1]["tool_calls"] == []
    assert captured == {
        "model": "anthropic/claude-sonnet-4.5",
        "provider_name": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
    }


def test_kimi_code_explicit_base_url_gets_coding_agent_headers(monkeypatch) -> None:
    import openai

    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            raise RuntimeError("stop before network")

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    with pytest.raises(RuntimeError, match="stop before network"):
        list(
            loop._stream_openai_compat(
                model="kimi-for-coding",
                system="",
                messages=[{"role": "user", "content": "hi"}],
                tool_schemas=[],
                config={"base_url": "https://api.kimi.com/coding/v1"},
                provider_name="kimi-code",
            )
        )

    assert captured["default_headers"]["User-Agent"] == "claude-code/1.0.0"
    assert captured["default_headers"]["X-Client-Name"] == "claude-code"
    assert captured["http_client"].headers["user-agent"] == "claude-code/1.0.0"
    captured["http_client"].close()
