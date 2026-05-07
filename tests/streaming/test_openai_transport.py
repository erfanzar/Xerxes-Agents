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
