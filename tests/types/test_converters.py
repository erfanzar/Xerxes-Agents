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
"""Tests for xerxes.types.converters module."""

import pytest
from xerxes.types.converters import (
    check_openai_fields_names,
    convert_openai_messages,
    convert_openai_tools,
    is_openai_field_name,
)
from xerxes.types.messages import AssistantMessage, SystemMessage, ToolMessage, UserMessage


class TestConvertOpenaiMessages:
    def test_user_message(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = convert_openai_messages(msgs)
        assert len(result) == 1
        assert isinstance(result[0], UserMessage)

    def test_system_message(self):
        msgs = [{"role": "system", "content": "You are helpful"}]
        result = convert_openai_messages(msgs)
        assert isinstance(result[0], SystemMessage)

    def test_assistant_message(self):
        msgs = [{"role": "assistant", "content": "Hi there"}]
        result = convert_openai_messages(msgs)
        assert isinstance(result[0], AssistantMessage)

    def test_tool_message(self):
        msgs = [{"role": "tool", "content": "result", "tool_call_id": "tc_1"}]
        result = convert_openai_messages(msgs)
        assert isinstance(result[0], ToolMessage)

    def test_unknown_role(self):
        msgs = [{"role": "unknown", "content": "test"}]
        with pytest.raises(ValueError, match="Unknown"):
            convert_openai_messages(msgs)

    def test_multiple_messages(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = convert_openai_messages(msgs)
        assert len(result) == 3


class TestConvertOpenaiTools:
    def test_basic_tool(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            }
        ]
        result = convert_openai_tools(tools)
        assert len(result) == 1
        assert result[0].function.name == "search"


class TestCheckOpenaiFieldNames:
    def test_valid_fields(self):
        check_openai_fields_names({"model", "messages"}, {"model", "messages"})

    def test_openai_valid_not_in_set(self):
        with pytest.raises(ValueError, match="OpenAI valid"):
            check_openai_fields_names({"custom"}, {"temperature"})

    def test_non_valid_params(self):
        with pytest.raises(ValueError, match="Non valid"):
            check_openai_fields_names({"custom"}, {"totally_fake_param"})


class TestIsOpenaiFieldName:
    def test_valid(self):
        assert is_openai_field_name("temperature") is True
        assert is_openai_field_name("model") is True

    def test_invalid(self):
        assert is_openai_field_name("not_a_field") is False
