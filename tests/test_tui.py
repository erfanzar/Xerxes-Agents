# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import asyncio

import xerxes.tui.app as tui_app_module
from xerxes import Agent, OperatorRuntimeConfig, PromptProfile, RuntimeFeaturesConfig, Xerxes
from xerxes.llms.base import BaseLLM, LLMConfig
from xerxes.tui import TextualLauncher, XerxesTUI, launch_tui, parse_command, preview_value
from xerxes.tui.app import ChatEntry
from xerxes.tui.terminal_config import TerminalConfigStore, TerminalProfile


def _chunk(
    *,
    content: str | None = None,
    buffered_content: str | None = None,
    reasoning_content: str | None = None,
    buffered_reasoning_content: str | None = None,
    function_calls: list[dict] | None = None,
    streaming_tool_calls: dict[int, dict] | list[dict] | None = None,
    is_final: bool = False,
) -> dict:
    buffered_content = buffered_content if buffered_content is not None else (content or "")
    return {
        "content": content,
        "buffered_content": buffered_content,
        "reasoning_content": reasoning_content,
        "buffered_reasoning_content": buffered_reasoning_content or "",
        "function_calls": function_calls or [],
        "tool_calls": None,
        "streaming_tool_calls": streaming_tool_calls,
        "raw_chunk": None,
        "is_final": is_final,
    }


class _FakeLLM(BaseLLM):
    def __init__(self, responses: list[list[dict]]):
        self.responses = list(responses)
        self.prompts: list[object] = []
        super().__init__(config=LLMConfig(model="fake-model"))

    def _initialize_client(self) -> None:
        self.client = object()

    async def generate_completion(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.responses.pop(0)

    def extract_content(self, response) -> str:
        return ""

    async def process_streaming_response(self, response, callback):
        output = ""
        for chunk in response:
            content = chunk.get("content")
            if content:
                callback(content, chunk)
                output += content
        return output

    def stream_completion(self, response, agent=None):
        yield from response

    async def astream_completion(self, response, agent=None):
        for chunk in response:
            yield chunk


def test_parse_command():
    assert parse_command("/profile none") == ("profile", "none")
    assert parse_command("/clear") == ("clear", "")
    assert parse_command("/") == ("", "")


def test_preview_value_truncates():
    assert preview_value("  hello  ") == "hello"
    long_text = "x" * 300
    assert preview_value(long_text, max_length=30).endswith("...")


def test_launch_tui_returns_launcher():
    xerxes = Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=True))
    agent = Agent(id="assistant", model="fake", instructions="Help", functions=[])
    xerxes.register_agent(agent)

    launcher = launch_tui(xerxes, agent)
    assert isinstance(launcher, TextualLauncher)
    assert xerxes.create_tui(agent) is not None


def test_xerxes_tui_advertises_enter_submit():
    assert all(binding.action != "submit_input" for binding in XerxesTUI.BINDINGS)
    assert "Enter to send" in XerxesTUI.DEFAULT_INPUT_PLACEHOLDER
    assert "Ctrl+Enter for newline" in XerxesTUI.DEFAULT_INPUT_PLACEHOLDER
    assert "Cmd+Enter" not in XerxesTUI.DEFAULT_INPUT_PLACEHOLDER


async def test_xerxes_tui_streaming_smoke():
    llm = _FakeLLM(
        responses=[
            [
                _chunk(reasoning_content="Inspect the request.", buffered_reasoning_content="Inspect the request."),
                _chunk(content="plain ", buffered_content="plain ", buffered_reasoning_content="Inspect the request."),
                _chunk(
                    content="answer",
                    buffered_content="plain answer",
                    buffered_reasoning_content="Inspect the request.",
                    is_final=True,
                ),
            ]
        ]
    )
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(enabled=True),
    )
    agent = Agent(id="assistant", model="fake-model", instructions="Test", functions=[])
    xerxes.register_agent(agent)

    app = XerxesTUI(executor=xerxes, agent=agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._current_prompt_profile() == PromptProfile.FULL

        app.action_cycle_profile()
        assert app._current_prompt_profile() == PromptProfile.COMPACT

        await app._stream_prompt("hello")
        assert [entry.role for entry in app.chat_history] == ["user", "assistant"]
        assert app.chat_history[-1].content == "plain answer"
        assert app.chat_history[-1].meta == "Inspect the request."
        assert app.chat_history[-1].streaming is False


async def test_xerxes_tui_accepts_multiline_composer_input():
    llm = _FakeLLM(
        responses=[
            [
                _chunk(content="received", buffered_content="received", is_final=True),
            ]
        ]
    )
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(enabled=True),
    )
    agent = Agent(id="assistant", model="fake-model", instructions="Test", functions=[])
    xerxes.register_agent(agent)

    app = XerxesTUI(executor=xerxes, agent=agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        composer = app._input_widget()
        composer.text = "line 1\nline 2"
        await app.action_submit_input()
        await pilot.pause()

    assert app.chat_history[0].role == "user"
    assert app.chat_history[0].content == "line 1\nline 2"
    assert app.chat_history[-1].content == "received"


async def test_xerxes_tui_submits_on_enter_key():
    llm = _FakeLLM(
        responses=[
            [
                _chunk(content="received", buffered_content="received", is_final=True),
            ]
        ]
    )
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(enabled=True),
    )
    agent = Agent(id="assistant", model="fake-model", instructions="Test", functions=[])
    xerxes.register_agent(agent)

    app = XerxesTUI(executor=xerxes, agent=agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        composer = app._input_widget()
        composer.text = "hello"
        composer.cursor_location = (0, len("hello"))
        await pilot.press("enter")
        await pilot.pause()

    assert app.chat_history[0].role == "user"
    assert app.chat_history[0].content == "hello"
    assert app.chat_history[-1].content == "received"


async def test_xerxes_tui_ctrl_enter_inserts_newline():
    xerxes = Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=True))
    agent = Agent(id="assistant", model="fake-model", instructions="Test", functions=[])
    xerxes.register_agent(agent)

    app = XerxesTUI(executor=xerxes, agent=agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        composer = app._input_widget()
        composer.text = "line 1"
        composer.cursor_location = (0, len("line 1"))
        await pilot.press("ctrl+enter")
        await pilot.pause()
        assert composer.text == "line 1\n"


async def test_xerxes_tui_passes_prior_turns_back_to_model():
    llm = _FakeLLM(
        responses=[
            [
                _chunk(content="first answer", buffered_content="first answer", is_final=True),
            ],
            [
                _chunk(content="second answer", buffered_content="second answer", is_final=True),
            ],
        ]
    )
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(enabled=True),
    )
    agent = Agent(id="assistant", model="fake-model", instructions="Test", functions=[])
    xerxes.register_agent(agent)

    app = XerxesTUI(executor=xerxes, agent=agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._stream_prompt("hello")
        await app._stream_prompt("follow up")

    assert len(llm.prompts) == 2
    second_prompt = llm.prompts[1]
    assert isinstance(second_prompt, list)
    assert second_prompt[1] == {"role": "user", "content": "hello"}
    assert second_prompt[2] == {"role": "assistant", "content": "first answer"}
    assert second_prompt[3] == {"role": "user", "content": "follow up"}


def test_xerxes_tui_conversation_messages_preserve_completed_tool_summary():
    xerxes = Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=True))
    agent = Agent(id="assistant", model="fake", instructions="Help", functions=[])
    xerxes.register_agent(agent)

    app = XerxesTUI(executor=xerxes, agent=agent)
    app.chat_history = [
        ChatEntry(role="user", content="search web about prism ml"),
        ChatEntry(
            role="tool",
            title="web.search_query ✓",
            meta='query="prism ml"',
            content='5 result(s); top="PrismML 1-Bit Bonsai LLM"',
        ),
        ChatEntry(role="assistant", content="The search results suggest there may be a PrismML page about it."),
    ]

    messages = app._conversation_messages().to_openai()["messages"]

    assert messages[0] == {"role": "user", "content": "search web about prism ml"}
    assert messages[1]["role"] == "assistant"
    assert "[Prior tool success] web.search_query" in messages[1]["content"]
    assert 'input: query="prism ml"' in messages[1]["content"]
    assert 'result: 5 result(s); top="PrismML 1-Bit Bonsai LLM"' in messages[1]["content"]
    assert messages[2] == {
        "role": "assistant",
        "content": "The search results suggest there may be a PrismML page about it.",
    }


async def test_xerxes_tui_tracks_tool_activity_inline():
    xerxes = Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=True))
    agent = Agent(id="assistant", model="fake", instructions="Help", functions=[])
    xerxes.register_agent(agent)

    app = XerxesTUI(executor=xerxes, agent=agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._start_tool_entry("call_1", "WriteFile", "1/1", agent_id="assistant")
        assert app.tool_activity["call_1"].status == "running"
        assert app.tool_activity["call_1"].progress == "1/1"
        # Tool cards now appear inline in chat history
        assert any(entry.role == "tool" and entry.key == "call_1" for entry in app.chat_history)

        app._complete_tool_entry("call_1", "WriteFile", "success", {"ok": True}, None, agent_id="assistant")
        assert app.tool_activity["call_1"].status == "success"
        assert "ok" in (app.tool_activity["call_1"].preview or "")
        # Tool card updated with result
        tool_entry = next(e for e in app.chat_history if e.role == "tool" and e.key == "call_1")
        assert not tool_entry.streaming
        assert "\u2713" in (tool_entry.title or "")


def test_xerxes_tui_tool_cards_keep_label_and_value_separated():
    xerxes = Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=True))
    agent = Agent(id="assistant", model="fake", instructions="Help", functions=[])
    xerxes.register_agent(agent)

    app = XerxesTUI(executor=xerxes, agent=agent)
    panel = app._build_message_renderable(
        ChatEntry(
            role="tool",
            title="web.search_query \u2713",
            meta='query="bonsai 1bit llm news"',
            content='5 result(s); top="Bonsai 1bit"',
        )
    )

    tool_lines = [renderable.plain for renderable in panel.renderable.renderables]

    assert tool_lines[0].startswith(" input ")
    assert tool_lines[1].startswith(" result ")
    assert "resultquery" not in tool_lines[1]
    assert "statusrunning" not in tool_lines[1]


def test_xerxes_tui_tool_result_summary_shows_search_fallback():
    xerxes = Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=True))
    agent = Agent(id="assistant", model="fake", instructions="Help", functions=[])
    xerxes.register_agent(agent)

    app = XerxesTUI(executor=xerxes, agent=agent)
    summary = app._summarize_tool_result(
        {
            "query": "latest OpenAI news",
            "search_type": "news",
            "effective_search_type": "text",
            "results": [{"title": "Fallback result", "url": "https://example.com/fallback"}],
        }
    )

    assert "fallback=news->text" in summary
    assert "1 result(s)" in summary


async def test_xerxes_tui_shows_compact_streamed_tool_cards():
    def lookup(query: str) -> dict:
        return {
            "query": query,
            "results": [
                {
                    "title": "OpenAI - Wikipedia",
                    "url": "https://en.wikipedia.org/wiki/OpenAI",
                }
            ],
        }

    llm = _FakeLLM(
        responses=[
            [
                _chunk(
                    streaming_tool_calls={
                        0: {
                            "id": "call_lookup",
                            "name": "lookup",
                            "arguments": '{"query":"OpenAI"}',
                        }
                    }
                ),
                _chunk(
                    function_calls=[
                        {
                            "id": "call_lookup",
                            "name": "lookup",
                            "arguments": {"query": "OpenAI"},
                        }
                    ],
                    is_final=True,
                ),
            ],
            [
                _chunk(content="Done", buffered_content="Done", is_final=True),
            ],
        ]
    )
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(enabled=True),
    )
    agent = Agent(id="assistant", model="fake-model", instructions="Test", functions=[lookup])
    xerxes.register_agent(agent)

    app = XerxesTUI(executor=xerxes, agent=agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._stream_prompt("search for OpenAI")

    tool_entries = [entry for entry in app.chat_history if entry.role == "tool" and entry.key == "call_lookup"]
    assert len(tool_entries) == 1
    tool_entry = tool_entries[0]
    assert tool_entry.meta == 'query="OpenAI"'
    assert "1 result(s)" in (tool_entry.content or "")
    assert "OpenAI - Wikipedia" in (tool_entry.content or "")
    assert "{" not in (tool_entry.content or "")
    assert "\u2713" in (tool_entry.title or "")


async def test_xerxes_tui_preserves_reasoning_across_reinvocations():
    def lookup(query: str) -> dict:
        return {"query": query, "results": [{"title": f"Result for {query}"}]}

    llm = _FakeLLM(
        responses=[
            [
                _chunk(
                    reasoning_content="Reasoning 1",
                    buffered_reasoning_content="Reasoning 1",
                    function_calls=[
                        {
                            "id": "call_1",
                            "name": "lookup",
                            "arguments": {"query": "first"},
                        }
                    ],
                    is_final=True,
                ),
            ],
            [
                _chunk(
                    reasoning_content="Reasoning 2",
                    buffered_reasoning_content="Reasoning 2",
                    function_calls=[
                        {
                            "id": "call_2",
                            "name": "lookup",
                            "arguments": {"query": "second"},
                        }
                    ],
                    is_final=True,
                ),
            ],
            [
                _chunk(
                    reasoning_content="Reasoning 3",
                    buffered_reasoning_content="Reasoning 3",
                ),
                _chunk(
                    content="Final answer",
                    buffered_content="Final answer",
                    buffered_reasoning_content="Reasoning 3",
                    is_final=True,
                ),
            ],
        ]
    )
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(enabled=True),
    )
    agent = Agent(id="assistant", model="fake-model", instructions="Test", functions=[lookup])
    xerxes.register_agent(agent)

    app = XerxesTUI(executor=xerxes, agent=agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._stream_prompt("start")

    assistant_entries = [entry for entry in app.chat_history if entry.role == "assistant"]
    tool_entries = [entry for entry in app.chat_history if entry.role == "tool"]

    assert [entry.meta for entry in assistant_entries] == ["Reasoning 1", "Reasoning 2", "Reasoning 3"]
    assert assistant_entries[-1].content == "Final answer"
    assert [entry.key for entry in tool_entries] == ["call_1", "call_2"]
    assert all(not entry.streaming for entry in assistant_entries)


async def test_xerxes_tui_accepts_pending_user_question_answer():
    llm = _FakeLLM(responses=[])
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            operator=OperatorRuntimeConfig(enabled=True),
        ),
    )
    agent = Agent(id="assistant", model="fake-model", instructions="Test", functions=[])
    xerxes.register_agent(agent)

    app = XerxesTUI(executor=xerxes, agent=agent)
    operator_state = app._operator_state()
    assert operator_state is not None

    task = asyncio.create_task(
        operator_state.user_prompt_manager.request(
            "Which mode should I use?",
            options=["scan", "grep"],
            allow_freeform=False,
        )
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        await asyncio.sleep(0)
        app._sync_pending_user_prompt()
        assert app.chat_history[-1].role == "question"
        assert "Which mode should I use?" in app.chat_history[-1].content

        app._submit_pending_user_answer("2")
        await pilot.pause()
        result = await task
        assert result["answer"] == "grep"
        assert app.chat_history[-1].role == "user"
        assert app.chat_history[-1].content == "grep"


async def test_xerxes_tui_footer_shows_token_stats():
    xerxes = Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=True))
    agent = Agent(id="assistant", model="gpt-4o-mini", instructions="Help", functions=[])
    xerxes.register_agent(agent)

    app = XerxesTUI(executor=xerxes, agent=agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._start_run_stats("hello world")
        app._update_run_stats("partial answer", "")
        assert app.prompt_tokens_used > 0
        assert app.output_tokens_used > 0
        assert app.tokens_per_second >= 0


async def test_xerxes_tui_hides_tool_markup_from_visible_reasoning_and_content():
    llm = _FakeLLM(
        responses=[
            [
                _chunk(
                    reasoning_content=(
                        "Inspect the request.\n<function=web.search_query><parameter=q>OpenAI</parameter></function>"
                    ),
                    buffered_reasoning_content=(
                        "Inspect the request.\n<function=web.search_query><parameter=q>OpenAI</parameter></function>"
                    ),
                ),
                _chunk(
                    content="Visible answer\n<function=web.search_query></function>",
                    buffered_content="Visible answer\n<function=web.search_query></function>",
                    buffered_reasoning_content=(
                        "Inspect the request.\n<function=web.search_query><parameter=q>OpenAI</parameter></function>"
                    ),
                    is_final=True,
                ),
            ]
        ]
    )
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(enabled=True),
    )
    agent = Agent(id="assistant", model="fake-model", instructions="Test", functions=[])
    xerxes.register_agent(agent)

    app = XerxesTUI(executor=xerxes, agent=agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._stream_prompt("hello")
        assert app.chat_history[-1].content == "Visible answer"
        assert app.chat_history[-1].meta == "Inspect the request."
        assert "<function=" not in app.chat_history[-1].content
        assert "<function=" not in (app.chat_history[-1].meta or "")


async def test_xerxes_tui_followup_prompt_includes_prior_tool_summary():
    llm = _FakeLLM(
        responses=[
            [
                _chunk(content="first answer", buffered_content="first answer", is_final=True),
            ],
            [
                _chunk(content="second answer", buffered_content="second answer", is_final=True),
            ],
        ]
    )
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(enabled=True),
    )
    agent = Agent(id="assistant", model="fake-model", instructions="Test", functions=[])
    xerxes.register_agent(agent)

    app = XerxesTUI(executor=xerxes, agent=agent)
    app.chat_history = [
        ChatEntry(role="user", content="search web about prism ml"),
        ChatEntry(
            role="tool",
            title="web.search_query ✓",
            meta='query="prism ml"',
            content='5 result(s); top="PrismML 1-Bit Bonsai LLM"',
        ),
        ChatEntry(role="assistant", content="The search results suggest there may be a PrismML page about it."),
    ]

    async with app.run_test() as pilot:
        await pilot.pause()
        await app._stream_prompt("u did a web search ;/")

    prompt = llm.prompts[0]
    assert isinstance(prompt, list)
    assert any(
        message["role"] == "assistant" and "[Prior tool success] web.search_query" in message["content"]
        for message in prompt
    )


async def test_xerxes_tui_model_change_persists(tmp_path):
    llm = _FakeLLM(responses=[[_chunk(content="ok", is_final=True)]])
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(enabled=True),
    )
    agent = Agent(id="assistant", model="fake-model", instructions="Test", functions=[])
    xerxes.register_agent(agent)

    store = TerminalConfigStore(tmp_path / "terminal_profiles.json")
    profile = TerminalProfile(
        name="lab",
        provider="openai",
        model="old-model",
        base_url="http://0.0.0.0:11556/v1/",
        api_key="sk-xxx",
        available_models=["old-model", "new-model"],
    )
    store.upsert_profile(profile)

    app = XerxesTUI(executor=xerxes, agent=agent, profile=profile, config_store=store)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._set_active_model("new-model")
        assert agent.model == "new-model"

    loaded = store.get_profile("lab")
    assert loaded is not None
    assert loaded.model == "new-model"


async def test_slash_hints_fuzzy_match_provider():
    xerxes = Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=True))
    agent = Agent(id="assistant", model="fake", instructions="Help", functions=[])
    xerxes.register_agent(agent)

    app = XerxesTUI(executor=xerxes, agent=agent)
    hints = app._match_slash_hints("/provide")
    assert hints
    assert hints[0].usage.startswith("/provider")

    suggestion = await app.command_suggester.get_suggestion("/provide")
    assert suggestion is not None
    assert suggestion.startswith("/provider")


async def test_slash_hints_show_provider_options():
    xerxes = Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=True))
    agent = Agent(id="assistant", model="fake", instructions="Help", functions=[])
    xerxes.register_agent(agent)

    app = XerxesTUI(executor=xerxes, agent=agent)
    hints = app._match_slash_hints("/provider o")
    assert [hint.usage for hint in hints] == ["/provider openai", "/provider ollama", "/provider oai"]


async def test_xerxes_tui_provider_command_persists_pending_provider(tmp_path, monkeypatch):
    llm = _FakeLLM(responses=[])
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(enabled=True),
    )
    agent = Agent(id="assistant", model="fake-model", instructions="Test", functions=[])
    xerxes.register_agent(agent)

    store = TerminalConfigStore(tmp_path / "terminal_profiles.json")
    profile = TerminalProfile(
        name="lab",
        provider="anthropic",
        model="claude-old",
        api_key=None,
        available_models=[],
    )
    store.upsert_profile(profile)

    calls: list[tuple[str, dict]] = []

    def fake_create_llm(provider: str, **kwargs):
        calls.append((provider, kwargs))
        fake = _FakeLLM(responses=[])
        fake.config.model = kwargs.get("model") or f"{provider}-default"
        fake.config.base_url = kwargs.get("base_url")
        fake.config.api_key = kwargs.get("api_key")
        return fake

    monkeypatch.setattr(tui_app_module, "create_llm", fake_create_llm)
    monkeypatch.setattr(
        tui_app_module,
        "discover_available_models",
        lambda *args, **kwargs: ["model-a", "model-b"],
    )

    app = XerxesTUI(executor=xerxes, agent=agent, profile=profile, config_store=store)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._handle_command("/provider openai")
        await pilot.pause()
        assert app._provider_name() == "openai"
        assert app._current_model_name() is None
        assert calls == []

    loaded = store.get_profile("lab")
    assert loaded is not None
    assert loaded.provider == "openai"
    assert loaded.model is None
    assert loaded.available_models == []


async def test_xerxes_tui_endpoint_after_pending_provider_uses_new_provider(tmp_path, monkeypatch):
    llm = _FakeLLM(responses=[])
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(enabled=True),
    )
    agent = Agent(id="assistant", model="fake-model", instructions="Test", functions=[])
    xerxes.register_agent(agent)

    store = TerminalConfigStore(tmp_path / "terminal_profiles.json")
    profile = TerminalProfile(
        name="lab",
        provider="anthropic",
        model="claude-old",
        api_key=None,
        available_models=[],
    )
    store.upsert_profile(profile)

    calls: list[tuple[str, dict]] = []

    def fake_create_llm(provider: str, **kwargs):
        calls.append((provider, kwargs))
        fake = _FakeLLM(responses=[])
        fake.config.model = kwargs.get("model") or f"{provider}-default"
        fake.config.base_url = kwargs.get("base_url")
        fake.config.api_key = kwargs.get("api_key")
        return fake

    monkeypatch.setattr(tui_app_module, "create_llm", fake_create_llm)
    monkeypatch.setattr(
        tui_app_module,
        "discover_available_models",
        lambda *args, **kwargs: ["model-a", "model-b"],
    )

    app = XerxesTUI(executor=xerxes, agent=agent, profile=profile, config_store=store)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._handle_command("/provider openai")
        await pilot.pause()
        assert app._handle_command("/endpoint http://0.0.0.0:11556/v1/")
        await pilot.pause()
        await pilot.pause()
        assert calls[-1][0] == "openai"
        assert app._provider_name() == "openai"
        assert app._current_model_name() == "model-a"

    loaded = store.get_profile("lab")
    assert loaded is not None
    assert loaded.provider == "openai"
    assert loaded.base_url == "http://0.0.0.0:11556/v1/"
    assert loaded.model == "model-a"
    assert loaded.available_models == ["model-a", "model-b"]


async def test_xerxes_tui_provider_alias_oai_switches_to_openai(tmp_path, monkeypatch):
    llm = _FakeLLM(responses=[])
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(enabled=True),
    )
    agent = Agent(id="assistant", model="fake-model", instructions="Test", functions=[])
    xerxes.register_agent(agent)

    store = TerminalConfigStore(tmp_path / "terminal_profiles.json")
    profile = TerminalProfile(
        name="lab",
        provider="ollama",
        model="llama3",
        base_url="http://0.0.0.0:11556/v1/",
        api_key="sk-test",
    )
    store.upsert_profile(profile)

    calls: list[tuple[str, dict]] = []

    def fake_create_llm(provider: str, **kwargs):
        calls.append((provider, kwargs))
        fake = _FakeLLM(responses=[])
        fake.config.model = kwargs.get("model") or f"{provider}-default"
        fake.config.base_url = kwargs.get("base_url")
        fake.config.api_key = kwargs.get("api_key")
        return fake

    monkeypatch.setattr(tui_app_module, "create_llm", fake_create_llm)
    monkeypatch.setattr(tui_app_module, "discover_available_models", lambda *args, **kwargs: ["qwen3-coder"])

    app = XerxesTUI(executor=xerxes, agent=agent, profile=profile, config_store=store)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._handle_command("/provider oai")
        await pilot.pause()
        await pilot.pause()
        assert calls[-1][0] == "openai"
        assert app._provider_name() == "openai"
        assert app._current_model_name() == "qwen3-coder"

    loaded = store.get_profile("lab")
    assert loaded is not None
    assert loaded.provider == "openai"


async def test_tui_formats_runtime_error_without_mozilla_link():
    xerxes = Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=True))
    agent = Agent(id="assistant", model="fake", instructions="Help", functions=[])
    xerxes.register_agent(agent)

    app = XerxesTUI(executor=xerxes, agent=agent)
    message = app._format_runtime_error(
        RuntimeError(
            "Client error '404 Not Found' for url 'http://x'\n"
            "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404"
        )
    )
    assert "developer.mozilla.org" not in message
    assert "404 Not Found" in message


async def test_xerxes_tui_endpoint_and_api_key_commands_persist(tmp_path, monkeypatch):
    llm = _FakeLLM(responses=[])
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(enabled=True),
    )
    agent = Agent(id="assistant", model="fake-model", instructions="Test", functions=[])
    xerxes.register_agent(agent)

    store = TerminalConfigStore(tmp_path / "terminal_profiles.json")
    profile = TerminalProfile(
        name="lab",
        provider="openai",
        model="old-model",
        base_url="http://old-endpoint/v1/",
        api_key="sk-old",
        available_models=["old-model"],
    )
    store.upsert_profile(profile)

    def fake_create_llm(provider: str, **kwargs):
        fake = _FakeLLM(responses=[])
        fake.config.model = kwargs.get("model") or "openai-default"
        fake.config.base_url = kwargs.get("base_url")
        fake.config.api_key = kwargs.get("api_key")
        return fake

    monkeypatch.setattr(tui_app_module, "create_llm", fake_create_llm)
    monkeypatch.setattr(tui_app_module, "discover_available_models", lambda *args, **kwargs: ["new-model"])

    app = XerxesTUI(executor=xerxes, agent=agent, profile=profile, config_store=store)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._handle_command("/endpoint http://0.0.0.0:11556/v1/")
        await pilot.pause()
        await pilot.pause()
        assert app._handle_command("/apikey sk-new-secret")
        await pilot.pause()
        await pilot.pause()
        assert app._base_url() == "http://0.0.0.0:11556/v1/"
        assert app._api_key() == "sk-new-secret"

    loaded = store.get_profile("lab")
    assert loaded is not None
    assert loaded.base_url == "http://0.0.0.0:11556/v1/"
    assert loaded.api_key == "sk-new-secret"
    assert loaded.available_models == ["new-model"]


async def test_slash_hints_show_model_options():
    xerxes = Xerxes(runtime_features=RuntimeFeaturesConfig(enabled=True))
    agent = Agent(id="assistant", model="fake", instructions="Help", functions=[])
    xerxes.register_agent(agent)

    app = XerxesTUI(executor=xerxes, agent=agent)
    app.available_models = ["qwen3-coder", "deepseek-r1"]
    hints = app._match_slash_hints("/model q")
    assert [hint.usage for hint in hints] == ["/model qwen3-coder"]


async def test_xerxes_tui_set_sampling_param_persists(tmp_path):
    llm = _FakeLLM(responses=[])
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(enabled=True),
    )
    agent = Agent(id="assistant", model="fake-model", instructions="Test", functions=[])
    xerxes.register_agent(agent)

    store = TerminalConfigStore(tmp_path / "terminal_profiles.json")
    profile = TerminalProfile(name="lab", provider="openai", model="fake-model")
    store.upsert_profile(profile)

    app = XerxesTUI(executor=xerxes, agent=agent, profile=profile, config_store=store)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._handle_command("/set temperature 0.35")
        await pilot.pause()
        assert app._handle_command("/set top-p 0.8")
        await pilot.pause()
        assert agent.temperature == 0.35
        assert agent.top_p == 0.8
        assert "temperature=0.35" in app.chat_history[-2].content
        assert "top_p=0.8" in app.chat_history[-1].content

    loaded = store.get_profile("lab")
    assert loaded is not None
    assert loaded.sampling_params["temperature"] == 0.35
    assert loaded.sampling_params["top_p"] == 0.8


async def test_xerxes_tui_reset_sampling_restores_defaults(tmp_path):
    llm = _FakeLLM(responses=[])
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(enabled=True),
    )
    agent = Agent(
        id="assistant",
        model="fake-model",
        instructions="Test",
        functions=[],
        temperature=1.2,
        max_tokens=512,
    )
    xerxes.register_agent(agent)

    store = TerminalConfigStore(tmp_path / "terminal_profiles.json")
    profile = TerminalProfile(
        name="lab",
        provider="openai",
        model="fake-model",
        sampling_params={"temperature": 1.2, "max_tokens": 512},
    )
    store.upsert_profile(profile)

    app = XerxesTUI(executor=xerxes, agent=agent, profile=profile, config_store=store)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._handle_command("/reset-sampling")
        await pilot.pause()
        assert agent.temperature == 0.7
        assert agent.max_tokens == 2048
        assert app.chat_history[-1].content == "Sampling reset to Agent defaults."

    loaded = store.get_profile("lab")
    assert loaded is not None
    assert loaded.sampling_params == {}


async def test_xerxes_tui_power_toggle_persists_and_lists_tools(tmp_path):
    llm = _FakeLLM(responses=[])
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            operator=OperatorRuntimeConfig(enabled=True),
        ),
    )
    agent = Agent(id="assistant", model="fake-model", instructions="Test", functions=[])
    xerxes.register_agent(agent)

    store = TerminalConfigStore(tmp_path / "terminal_profiles.json")
    profile = TerminalProfile(name="lab", provider="openai", model="fake-model")
    store.upsert_profile(profile)

    app = XerxesTUI(executor=xerxes, agent=agent, profile=profile, config_store=store)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._handle_command("/power on")
        await pilot.pause()
        assert app._power_tools_enabled() is True
        assert app._handle_command("/tools")
        await pilot.pause()
        assert "web.time" in app.chat_history[-1].content
        assert "exec_command" in app.chat_history[-1].content

    loaded = store.get_profile("lab")
    assert loaded is not None
    assert loaded.power_tools_enabled is True


async def test_xerxes_tui_plans_command_shows_current_plan():
    llm = _FakeLLM(responses=[])
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(
            enabled=True,
            operator=OperatorRuntimeConfig(enabled=True),
        ),
    )
    agent = Agent(id="assistant", model="fake-model", instructions="Test", functions=[])
    xerxes.register_agent(agent)

    app = XerxesTUI(executor=xerxes, agent=agent)
    operator_state = app._operator_state()
    assert operator_state is not None
    operator_state.plan_manager.update(
        "Work through the operator tasks.",
        [{"step": "Wire operator tools", "status": "in_progress"}],
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._handle_command("/plans")
        await pilot.pause()
        assert "Plan revision 1:" in app.chat_history[-1].content
        assert "Wire operator tools" in app.chat_history[-1].content
