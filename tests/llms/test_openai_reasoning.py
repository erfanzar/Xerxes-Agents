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
from types import SimpleNamespace

from xerxes.llms.openai import OpenAILLM


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self):
        self.chat = _FakeChat()


class _FakeAsyncCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs


class _FakeAsyncChat:
    def __init__(self):
        self.completions = _FakeAsyncCompletions()


class _FakeAsyncClient:
    def __init__(self):
        self.chat = _FakeAsyncChat()


def _make_llm() -> OpenAILLM:
    return OpenAILLM(model="gpt-4o-mini", client=object())


def _make_chunk(
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    model_extra: dict | None = None,
):
    delta = SimpleNamespace(
        content=content,
        tool_calls=None,
        model_extra=model_extra or {},
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def test_extract_reasoning_content_from_message_model_extra():
    llm = _make_llm()
    message = SimpleNamespace(content=None, model_extra={"delta_reasoning": "Inspect the request first."})
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])

    assert llm.extract_reasoning_content(response) == "Inspect the request first."


def test_stream_completion_extracts_delta_reasoning_from_model_extra():
    llm = _make_llm()
    chunks = list(
        llm.stream_completion(
            [
                _make_chunk(model_extra={"delta_reasoning": "Plan carefully. "}),
                _make_chunk(content="Final answer.", finish_reason="stop"),
            ]
        )
    )

    assert chunks[0]["reasoning_content"] == "Plan carefully. "
    assert chunks[0]["buffered_reasoning_content"] == "Plan carefully. "
    assert chunks[1]["content"] == "Final answer."
    assert chunks[1]["buffered_reasoning_content"] == "Plan carefully. "


def test_stream_completion_extracts_reasoning_from_openai_response_events():
    llm = _make_llm()
    event = SimpleNamespace(type="response.reasoning_summary_text.delta", delta="Compare candidate approaches.")

    chunks = list(llm.stream_completion([event]))

    assert len(chunks) == 1
    assert chunks[0]["reasoning_content"] == "Compare candidate approaches."
    assert chunks[0]["buffered_reasoning_content"] == "Compare candidate approaches."


async def test_generate_completion_sends_compat_sampling_params_to_extra_body_for_custom_base_url():
    client = _FakeClient()
    llm = OpenAILLM(model="qwen", base_url="http://localhost:11556/v1/", client=client)

    result = await llm.generate_completion(
        "Hello",
        stream=False,
        top_k=20,
        min_p=0.05,
        repetition_penalty=1.15,
        presence_penalty=0.3,
        frequency_penalty=0.4,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )

    assert result["presence_penalty"] == 0.3
    assert result["frequency_penalty"] == 0.4
    assert "top_k" not in result
    assert "min_p" not in result
    assert "repetition_penalty" not in result
    assert result["extra_body"]["top_k"] == 20
    assert result["extra_body"]["min_p"] == 0.05
    assert result["extra_body"]["repetition_penalty"] == 1.15
    assert result["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}


async def test_generate_completion_keeps_compat_sampling_params_off_official_openai_requests():
    client = _FakeClient()
    llm = OpenAILLM(model="gpt-4o-mini", client=client)

    result = await llm.generate_completion(
        "Hello",
        stream=False,
        top_k=20,
        min_p=0.05,
        repetition_penalty=1.15,
    )

    assert "extra_body" not in result
    assert "top_k" not in result
    assert "min_p" not in result
    assert "repetition_penalty" not in result


async def test_generate_completion_prefers_async_client_when_available():
    sync_client = _FakeClient()
    async_client = _FakeAsyncClient()
    llm = OpenAILLM(model="gpt-4o-mini", client=sync_client, async_client=async_client)

    result = await llm.generate_completion("Hello", stream=False)

    assert result["model"] == "gpt-4o-mini"
    assert len(async_client.chat.completions.calls) == 1
    assert sync_client.chat.completions.calls == []


async def test_generate_completion_wraps_sync_stream_as_async_iterator():
    class _StreamCompletions:
        def create(self, **kwargs):
            return [kwargs]

    class _StreamChat:
        def __init__(self):
            self.completions = _StreamCompletions()

    class _StreamClient:
        def __init__(self):
            self.chat = _StreamChat()

    client = _StreamClient()
    llm = OpenAILLM(model="gpt-4o-mini", client=client)

    stream = await llm.generate_completion("Hello", stream=True)

    assert hasattr(stream, "__aiter__")
    collected = []
    async for chunk in stream:
        collected.append(chunk)
    assert len(collected) == 1
    assert collected[0]["model"] == "gpt-4o-mini"
