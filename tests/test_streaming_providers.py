# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from xerxes_agent.llms.anthropic import AnthropicLLM
from xerxes_agent.llms.ollama import OllamaLLM


async def test_ollama_generate_completion_stream_returns_async_iterator(monkeypatch):
    monkeypatch.setattr(OllamaLLM, "fetch_model_info", lambda self: {})

    llm = OllamaLLM(model="llama3", base_url="http://localhost:11434")

    stream = await llm.generate_completion("hello", stream=True)

    assert hasattr(stream, "__aiter__")
    await llm.close()


async def test_anthropic_generate_completion_stream_returns_async_iterator(monkeypatch):
    monkeypatch.setattr(AnthropicLLM, "fetch_model_info", lambda self: {})

    llm = AnthropicLLM(model="claude-3-haiku-20240307", api_key="sk-ant-test")

    stream = await llm.generate_completion("hello", stream=True)

    assert hasattr(stream, "__aiter__")
    await llm.close()
