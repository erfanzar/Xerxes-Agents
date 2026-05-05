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
"""Tests for xerxes.types.messages module — focus on uncovered branches."""

from xerxes.types.messages import (
    AssistantMessage,
    MessagesHistory,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from xerxes.types.tool_calls import FunctionCall, ToolCall


class TestMessagesHistory:
    def test_creation(self):
        msgs = MessagesHistory(
            messages=[
                SystemMessage(content="system"),
                UserMessage(content="hello"),
            ]
        )
        assert len(msgs.messages) == 2

    def test_from_openai(self):
        openai_msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        history = MessagesHistory.from_openai(openai_msgs)
        assert len(history.messages) == 3

    def test_make_instruction_prompt_basic(self):
        msgs = MessagesHistory(
            messages=[
                SystemMessage(content="You are helpful"),
                UserMessage(content="Hello"),
                AssistantMessage(content="Hi there"),
            ]
        )
        prompt = msgs.make_instruction_prompt()
        assert "Instruction" in prompt
        assert "Hello" in prompt

    def test_make_instruction_prompt_no_system(self):
        msgs = MessagesHistory(
            messages=[
                UserMessage(content="Hello"),
            ]
        )
        prompt = msgs.make_instruction_prompt()
        assert "No system prompt" in prompt

    def test_make_instruction_prompt_with_tool_message(self):
        msgs = MessagesHistory(
            messages=[
                UserMessage(content="Do something"),
                ToolMessage(content="Result data", tool_call_id="tc_1"),
            ]
        )
        prompt = msgs.make_instruction_prompt()
        assert "Result data" in prompt or "Tool Result" in prompt

    def test_make_instruction_prompt_no_last_turn(self):
        msgs = MessagesHistory(
            messages=[
                UserMessage(content="Hello"),
            ]
        )
        prompt = msgs.make_instruction_prompt(mention_last_turn=False)
        assert "Last Message" not in prompt

    def test_make_instruction_prompt_custom_holder(self):
        msgs = MessagesHistory(
            messages=[
                UserMessage(content="Hello"),
            ]
        )
        prompt = msgs.make_instruction_prompt(conversation_name_holder="Chat")
        assert "Chat" in prompt


class TestUserMessage:
    def test_from_openai_string_content(self):
        msg = UserMessage.from_openai({"role": "user", "content": "Hello"})
        assert msg.content == "Hello"

    def test_from_openai_list_content(self):
        msg = UserMessage.from_openai(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hi"},
                ],
            }
        )
        assert msg.content is not None


class TestAssistantMessage:
    def test_from_openai(self):
        msg = AssistantMessage.from_openai({"role": "assistant", "content": "Hi"})
        assert msg.content == "Hi"

    def test_from_openai_with_tool_calls(self):
        msg = AssistantMessage.from_openai(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "tc_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
            }
        )
        assert msg.tool_calls is not None

    def test_to_openai_includes_empty_content_for_tool_calls(self):
        msg = AssistantMessage(
            content=None,
            tool_calls=[ToolCall(id="tc_1", function=FunctionCall(name="search", arguments="{}"))],
        )

        payload = msg.to_openai()

        assert payload["content"] == ""
        assert payload["tool_calls"][0]["id"] == "tc_1"


class TestSystemMessage:
    def test_from_openai(self):
        msg = SystemMessage.from_openai({"role": "system", "content": "Be helpful"})
        assert msg.content == "Be helpful"


class TestToolMessage:
    def test_from_openai(self):
        msg = ToolMessage.from_openai({"role": "tool", "content": "result", "tool_call_id": "tc_1"})
        assert msg.content == "result"
        assert msg.tool_call_id == "tc_1"
