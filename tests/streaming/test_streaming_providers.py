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

from xerxes.llms.anthropic import AnthropicLLM
from xerxes.llms.ollama import OllamaLLM


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
