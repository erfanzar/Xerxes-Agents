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

import pytest
from xerxes import Agent, Xerxes
from xerxes.llms.base import BaseLLM, LLMConfig
from xerxes.types import (
    AssistantMessage,
    ExecutionResult,
    ExecutionStatus,
    MessagesHistory,
    RequestFunctionCall,
    SystemMessage,
    ToolMessage,
    UserMessage,
)


def _tool_example(command: str) -> dict[str, str]:
    return {"command": command}


def _web_search_query(q: str, search_type: str = "text", n_results: int = 5) -> dict[str, object]:
    return {
        "query": q,
        "search_type": search_type,
        "n_results": n_results,
        "results": [{"title": "OpenAI latest", "url": "https://example.com/openai"}],
    }


_web_search_query.__xerxes_schema__ = {
    "name": "web.search_query",
    "description": "Search the public web through DuckDuckGo and return compact result dictionaries.",
    "parameters": {
        "type": "object",
        "properties": {
            "q": {"type": "string", "description": "Search query text."},
            "search_type": {"type": "string", "description": "Search vertical such as text or news."},
            "n_results": {"type": "integer", "description": "Maximum number of results to return."},
        },
        "required": ["q"],
    },
}


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
    xerxes = Xerxes()

    messages = xerxes.manage_messages(agent=agent, prompt="List files in the current directory.")
    system_message = messages.messages[0]

    assert "Do not use functions for greetings, simple conversation" in system_message.content
    assert "If the user explicitly asks to search, look up, browse, or find something on the web" in (
        system_message.content
    )
    assert "If the user gives a generic follow-up like `search the web`, `look it up`, or `find it`" in (
        system_message.content
    )
    assert "do not say that you cannot browse, search the web, or access current information" in (system_message.content)
    assert "Search-result snippets are not verified facts" in system_message.content
    assert "Do not simulate tool calls or wrap normal answers in tool/XML markup." in system_message.content
    assert "After a function returns a result, use that result to continue the task and answer the user." in (
        system_message.content
    )
    assert "Do not repeat the same function call with the same arguments" in system_message.content


def test_build_reinvoke_messages_appends_followup_instruction_after_tool_results():
    agent = Agent(model="gpt-4o-mini", functions=[_tool_example])
    xerxes = Xerxes()
    original_messages = xerxes.manage_messages(agent=agent, prompt="List files in the current directory.")

    function_calls = [RequestFunctionCall(name="_tool_example", arguments={"command": "ls"}, id="call_1")]
    results = [ExecutionResult(status=ExecutionStatus.SUCCESS, result={"stdout": "README.md\nsrc\n", "stderr": ""})]

    updated_messages = xerxes._build_reinvoke_messages(
        original_messages=original_messages,
        assistant_content="",
        function_calls=function_calls,
        results=results,
    )

    assert isinstance(updated_messages.messages[-3], AssistantMessage)
    assert isinstance(updated_messages.messages[-2], ToolMessage)
    assert isinstance(updated_messages.messages[-1], UserMessage)
    assert "Use the function results above to continue the task." in updated_messages.messages[-1].content
    assert "do not claim you cannot browse or access current information" in updated_messages.messages[-1].content
    assert "Treat search-result snippets as leads rather than verified facts" in updated_messages.messages[-1].content
    assert updated_messages.messages[-2].tool_call_id == "call_1"


def test_manage_messages_dedupes_system_prompt_when_reinvoking():
    agent = Agent(model="gpt-4o-mini", functions=[_tool_example])
    xerxes = Xerxes()
    original_messages = xerxes.manage_messages(agent=agent, prompt="List files in the current directory.")

    function_calls = [RequestFunctionCall(name="_tool_example", arguments={"command": "ls"}, id="call_1")]
    results = [ExecutionResult(status=ExecutionStatus.SUCCESS, result={"stdout": "README.md\nsrc\n", "stderr": ""})]

    updated_messages = xerxes._build_reinvoke_messages(
        original_messages=original_messages,
        assistant_content="",
        function_calls=function_calls,
        results=results,
    )
    reinvoked_messages = xerxes.manage_messages(agent=agent, messages=updated_messages)

    system_messages = [message for message in reinvoked_messages.messages if isinstance(message, SystemMessage)]

    assert len(system_messages) == 1
    assert reinvoked_messages.messages[0] is system_messages[0]
    assert [message.role for message in reinvoked_messages.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert system_messages[0].content.count("After a function returns a result") == 1


def test_extract_function_calls_parses_tagged_function_markup():
    agent = Agent(model="gpt-4o-mini", functions=[_web_search_query])
    xerxes = Xerxes()

    content = """
<function=web.search_query>
<parameter=q>
latest OpenAI news
</parameter>
<parameter=search_type>
news
</parameter>
</function>
""".strip()

    function_calls = xerxes._extract_function_calls(content, agent, None)

    assert len(function_calls) == 1
    assert function_calls[0].name == "web.search_query"
    assert function_calls[0].arguments == {"q": "latest OpenAI news", "search_type": "news"}
    assert xerxes._remove_function_calls_from_content(content) == ""


@pytest.mark.asyncio
async def test_explicit_web_search_request_is_left_to_model_tool_choice():
    llm = _FakeLLM(
        responses=[
            [
                _chunk(
                    function_calls=[
                        {
                            "id": "call_search",
                            "name": "web.search_query",
                            "arguments": {"q": "latest OpenAI news", "search_type": "news"},
                        }
                    ],
                    is_final=True,
                )
            ],
            [[_chunk(content="Here are the current web search results.", is_final=True)]][0],
        ]
    )
    xerxes = Xerxes(llm=llm)
    agent = Agent(id="assistant", model="fake-model", instructions="Test", functions=[_web_search_query])
    xerxes.register_agent(agent)

    result = await xerxes.create_response(
        prompt="Search the web for the latest OpenAI news.",
        agent_id=agent,
        stream=False,
    )

    assert len(llm.calls) == 2
    assert result.content == "Here are the current web search results."
    assert result.function_calls[0].name == "web.search_query"
    first_prompt_messages = llm.calls[0]["prompt"]
    assert not any(message["role"] == "tool" for message in first_prompt_messages)
    tool_message = next(message for message in llm.calls[1]["prompt"] if message["role"] == "tool")
    assert "latest OpenAI news" in str(tool_message["content"])


@pytest.mark.asyncio
async def test_generic_followup_web_search_relies_on_prompt_history_instead_of_forcing_tool():
    llm = _FakeLLM(responses=[[_chunk(content="Tell me what to search for.", is_final=True)]])
    xerxes = Xerxes(llm=llm)
    agent = Agent(id="assistant", model="fake-model", instructions="Test", functions=[_web_search_query])
    xerxes.register_agent(agent)

    history = MessagesHistory(
        messages=[
            UserMessage(content="read about bonsai 1bit llm from prism-ml"),
            AssistantMessage(content="I do not know enough about that yet."),
        ]
    )

    await xerxes.create_response(
        prompt="search the web",
        messages=history,
        agent_id=agent,
        stream=False,
    )

    assert len(llm.calls) == 1
    prompt_messages = llm.calls[0]["prompt"]
    assert prompt_messages[1] == {"role": "user", "content": "read about bonsai 1bit llm from prism-ml"}
    assert prompt_messages[2] == {"role": "assistant", "content": "I do not know enough about that yet."}
    assert prompt_messages[3] == {"role": "user", "content": "search the web"}
    assert not any(message["role"] == "tool" for message in prompt_messages)
    assert "generic follow-up like `search the web`, `look it up`, or `find it`" in prompt_messages[0]["content"]


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
    xerxes = Xerxes(llm=llm)
    agent = Agent(id="assistant", model="fake-model", instructions="Test", functions=[_tool_example])
    xerxes.register_agent(agent)

    result = await xerxes.create_response(prompt="re do it again", agent_id=agent, stream=False)

    assert result.content == "There is no prior task to redo."
    assert result.function_calls == []
    assert len(llm.calls) == 1
