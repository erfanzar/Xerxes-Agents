# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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

import pytest

from calute import Agent, Calute
from calute.llms.base import BaseLLM, LLMConfig
from calute.types import (
    AssistantMessage,
    ExecutionResult,
    ExecutionStatus,
    RequestFunctionCall,
    ToolMessage,
    UserMessage,
)


def _tool_example(command: str) -> dict[str, str]:
    return {"command": command}


def _chunk(
    *,
    content: str | None = None,
    buffered_content: str | None = None,
    function_calls: list[dict] | None = None,
    is_final: bool = False,
) -> dict:
    buffered_content = buffered_content if buffered_content is not None else (content or "")
    return {
        "content": content,
        "buffered_content": buffered_content,
        "reasoning_content": None,
        "buffered_reasoning_content": "",
        "function_calls": function_calls or [],
        "tool_calls": None,
        "streaming_tool_calls": None,
        "raw_chunk": None,
        "is_final": is_final,
    }


class _FakeLLM(BaseLLM):
    def __init__(self, responses: list[list[dict]]):
        self.responses = list(responses)
        self.calls: list[dict] = []
        super().__init__(config=LLMConfig(model="fake-model"))

    def _initialize_client(self) -> None:
        self.client = object()

    async def generate_completion(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return self.responses.pop(0)

    def extract_content(self, response) -> str:
        return ""

    async def process_streaming_response(self, response, callback):
        output = ""
        for chunk in response:
            if chunk.get("content"):
                callback(chunk["content"], chunk)
                output += chunk["content"]
        return output

    def stream_completion(self, response, agent=None):
        yield from response

    async def astream_completion(self, response, agent=None):
        for chunk in response:
            yield chunk


def test_manage_messages_adds_post_tool_rules_for_native_tool_mode():
    agent = Agent(model="gpt-4o-mini", functions=[_tool_example])
    calute = Calute()

    messages = calute.manage_messages(agent=agent, prompt="List files in the current directory.")
    system_message = messages.messages[0]

    assert "After a function returns a result, use that result to continue the task and answer the user." in (
        system_message.content
    )
    assert "Do not repeat the same function call with the same arguments" in system_message.content


def test_build_reinvoke_messages_appends_followup_instruction_after_tool_results():
    agent = Agent(model="gpt-4o-mini", functions=[_tool_example])
    calute = Calute()
    original_messages = calute.manage_messages(agent=agent, prompt="List files in the current directory.")

    function_calls = [RequestFunctionCall(name="_tool_example", arguments={"command": "ls"}, id="call_1")]
    results = [ExecutionResult(status=ExecutionStatus.SUCCESS, result={"stdout": "README.md\nsrc\n", "stderr": ""})]

    updated_messages = calute._build_reinvoke_messages(
        original_messages=original_messages,
        assistant_content="",
        function_calls=function_calls,
        results=results,
    )

    assert isinstance(updated_messages.messages[-3], AssistantMessage)
    assert isinstance(updated_messages.messages[-2], ToolMessage)
    assert isinstance(updated_messages.messages[-1], UserMessage)
    assert "Use the function results above to continue the task." in updated_messages.messages[-1].content
    assert updated_messages.messages[-2].tool_call_id == "call_1"


@pytest.mark.asyncio
async def test_unknown_provider_tool_calls_are_ignored_instead_of_reinvoking():
    llm = _FakeLLM(
        responses=[
            [
                _chunk(
                    content="There is no prior task to redo.",
                    function_calls=[
                        {
                            "id": "bad_1",
                            "name": "user_request_reinterpretation",
                            "arguments": {"prompt": "re do it again"},
                        }
                    ],
                    is_final=True,
                )
            ]
        ]
    )
    calute = Calute(llm=llm)
    agent = Agent(id="assistant", model="fake-model", instructions="Test", functions=[_tool_example])
    calute.register_agent(agent)

    result = await calute.create_response(prompt="re do it again", agent_id=agent, stream=False)

    assert result.content == "There is no prior task to redo."
    assert result.function_calls == []
    assert len(llm.calls) == 1
