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
"""Tests for xerxes.api_server models and converters."""

import pytest
from xerxes.api_server.converters import MessageConverter
from xerxes.api_server.models import HealthResponse, ModelInfo, ModelsResponse
from xerxes.types.oai_protocols import ChatMessage


class TestModelInfo:
    def test_basic(self):
        m = ModelInfo(id="gpt-4", created=1234567890)
        assert m.id == "gpt-4"
        assert m.object == "model"
        assert m.owned_by == "xerxes"

    def test_custom_owner(self):
        m = ModelInfo(id="test", created=0, owned_by="custom")
        assert m.owned_by == "custom"


class TestModelsResponse:
    def test_basic(self):
        models = [ModelInfo(id="m1", created=1), ModelInfo(id="m2", created=2)]
        resp = ModelsResponse(data=models)
        assert resp.object == "list"
        assert len(resp.data) == 2


class TestHealthResponse:
    def test_basic(self):
        h = HealthResponse(status="healthy", agents=3)
        assert h.status == "healthy"
        assert h.agents == 3


class TestMessageConverter:
    def test_user_message(self):
        msgs = [ChatMessage(role="user", content="Hello")]
        result = MessageConverter.convert_openai_to_xerxes(msgs)
        assert len(result.messages) == 1

    def test_system_message(self):
        msgs = [ChatMessage(role="system", content="You are helpful")]
        result = MessageConverter.convert_openai_to_xerxes(msgs)
        assert len(result.messages) == 1

    def test_assistant_message(self):
        msgs = [ChatMessage(role="assistant", content="Hi there")]
        result = MessageConverter.convert_openai_to_xerxes(msgs)
        assert len(result.messages) == 1

    def test_unknown_role(self):
        msgs = [ChatMessage(role="unknown", content="test")]
        with pytest.raises(ValueError, match="Unknown"):
            MessageConverter.convert_openai_to_xerxes(msgs)

    def test_multiple_messages(self):
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
        ]
        result = MessageConverter.convert_openai_to_xerxes(msgs)
        assert len(result.messages) == 3

    def test_empty_content(self):
        msgs = [ChatMessage(role="user", content="")]
        result = MessageConverter.convert_openai_to_xerxes(msgs)
        assert len(result.messages) == 1
