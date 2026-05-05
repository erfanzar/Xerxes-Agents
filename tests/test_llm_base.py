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
"""Tests for xerxes.llms.base module."""

import pytest
from xerxes.llms.base import BaseLLM, LLMConfig


class TestLLMConfig:
    def test_defaults(self):
        config = LLMConfig(model="gpt-4")
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.stream is False

    def test_custom(self):
        config = LLMConfig(model="claude-3", temperature=0.5, max_tokens=4096, stream=True)
        assert config.temperature == 0.5
        assert config.max_tokens == 4096
        assert config.stream is True

    def test_extra_params(self):
        config = LLMConfig(model="test", extra_params={"custom": "value"})
        assert config.extra_params["custom"] == "value"


class ConcreteLLM(BaseLLM):
    """Concrete implementation for testing."""

    def _initialize_client(self):
        self.client = "mock_client"

    async def generate_completion(self, prompt, **kwargs):
        return {"content": "response"}

    def extract_content(self, response):
        return response.get("content", "")

    async def process_streaming_response(self, response, callback):
        return "streamed"

    def stream_completion(self, response, agent=None):
        yield {"content": "chunk", "is_final": True}

    async def astream_completion(self, response, agent=None):
        yield {"content": "async_chunk", "is_final": True}


class TestBaseLLM:
    def test_init_with_config(self):
        config = LLMConfig(model="test-model")
        llm = ConcreteLLM(config)
        assert llm.config.model == "test-model"
        assert llm.client == "mock_client"

    def test_init_default_config(self):
        llm = ConcreteLLM()
        assert llm.config.model == "default"

    def test_validate_config(self):
        config = LLMConfig(model="test")
        llm = ConcreteLLM(config)
        llm.validate_config()

    def test_validate_config_empty_model(self):
        config = LLMConfig(model="")
        llm = ConcreteLLM(config)
        with pytest.raises(ValueError, match="Model name"):
            llm.validate_config()

    def test_validate_config_bad_temperature(self):
        config = LLMConfig(model="test", temperature=3.0)
        llm = ConcreteLLM(config)
        with pytest.raises(ValueError, match="Temperature"):
            llm.validate_config()

    def test_validate_config_bad_max_tokens(self):
        config = LLMConfig(model="test", max_tokens=-1)
        llm = ConcreteLLM(config)
        with pytest.raises(ValueError, match="max_tokens"):
            llm.validate_config()

    def test_validate_config_bad_top_p(self):
        config = LLMConfig(model="test", top_p=0.0)
        llm = ConcreteLLM(config)
        with pytest.raises(ValueError, match="top_p"):
            llm.validate_config()

    def test_parse_tool_calls_default(self):
        llm = ConcreteLLM()
        assert llm.parse_tool_calls(None) == []

    def test_format_messages_no_system(self):
        llm = ConcreteLLM()
        msgs = [{"role": "user", "content": "hi"}]
        result = llm.format_messages(msgs)
        assert len(result) == 1

    def test_format_messages_with_system(self):
        llm = ConcreteLLM()
        msgs = [{"role": "user", "content": "hi"}]
        result = llm.format_messages(msgs, system_prompt="Be helpful")
        assert len(result) == 2
        assert result[0]["role"] == "system"

    def test_fetch_model_info(self):
        llm = ConcreteLLM()
        assert llm.fetch_model_info() == {}

    def test_auto_fetch_model_info(self):
        llm = ConcreteLLM()
        llm._auto_fetch_model_info()

    def test_get_model_info(self):
        config = LLMConfig(model="test-model")
        llm = ConcreteLLM(config)
        info = llm.get_model_info()
        assert info["model"] == "test-model"
        assert "provider" in info

    def test_repr(self):
        config = LLMConfig(model="test")
        llm = ConcreteLLM(config)
        repr_str = repr(llm)
        assert "test" in repr_str

    async def test_context_manager(self):
        config = LLMConfig(model="test")
        async with ConcreteLLM(config) as llm:
            assert llm.config.model == "test"

    def test_extract_content(self):
        llm = ConcreteLLM()
        assert llm.extract_content({"content": "hello"}) == "hello"
