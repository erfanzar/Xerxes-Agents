# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import argparse
from types import SimpleNamespace

from calute.tui.cli import (
    DEFAULT_INSTRUCTIONS,
    _resolve_profile,
    build_default_agent,
    detect_model,
    detect_provider,
    looks_openai_compatible_endpoint,
    main,
)
from calute.tui.terminal_config import TerminalConfigStore, TerminalProfile


def test_detect_provider_prefers_explicit_env():
    env = {
        "CALUTE_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-test",
    }
    assert detect_provider(env) == "openai"


def test_detect_provider_falls_back_to_ollama():
    assert detect_provider({}) == "ollama"


def test_detect_model_prefers_generic_env():
    env = {
        "CALUTE_MODEL": "gpt-4.1-mini",
        "OPENAI_MODEL": "ignored",
    }
    assert detect_model("openai", env) == "gpt-4.1-mini"


def test_looks_openai_compatible_endpoint_detects_v1():
    assert looks_openai_compatible_endpoint("http://35.193.63.250:11556/v1/")
    assert not looks_openai_compatible_endpoint("http://localhost:11434")


def test_build_default_agent_includes_tools():
    agent = build_default_agent(model="gpt-4o-mini")
    assert agent.model == "gpt-4o-mini"
    assert len(agent.functions) >= 4
    available = set(agent.get_available_functions())
    assert "ReadFile" in available
    assert "WriteFile" in available
    assert "ListDir" in available
    assert "ExecutePythonCode" in available
    assert "DuckDuckGoSearch" not in available
    assert "ExecuteShell" not in available


def test_build_default_agent_applies_sampling_params():
    agent = build_default_agent(
        model="gpt-4o-mini",
        sampling_params={"temperature": 0.2, "top_p": 0.8, "max_tokens": 4096},
    )
    assert agent.temperature == 0.2
    assert agent.top_p == 0.8
    assert agent.max_tokens == 4096


def test_default_instructions_discourage_fake_tool_markup():
    assert "Answer directly when the request can be handled from the conversation alone." in DEFAULT_INSTRUCTIONS
    assert "Use tools sparingly" in DEFAULT_INSTRUCTIONS
    assert "If the user explicitly asks you to search/look up/browse the web" in DEFAULT_INSTRUCTIONS
    assert (
        "If the user gives a generic follow-up like `search the web`, `look it up`, or `find it`" in DEFAULT_INSTRUCTIONS
    )
    assert "Read tool descriptions and parameter docs carefully" in DEFAULT_INSTRUCTIONS
    assert "do not claim that you cannot browse or access current information" in DEFAULT_INSTRUCTIONS
    assert "Treat search-result snippets as leads rather than verified facts" in DEFAULT_INSTRUCTIONS
    assert "Never simulate tool calls or emit tool/XML wrappers" in DEFAULT_INSTRUCTIONS
    assert "Put the final answer in plain assistant text" in DEFAULT_INSTRUCTIONS


def test_main_launches_tui(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class _Launcher:
        def launch(self):
            captured["launched"] = True

    monkeypatch.setattr("calute.tui.cli.create_llm", lambda provider, **kwargs: object())
    monkeypatch.setattr("calute.tui.cli.discover_available_models", lambda *args, **kwargs: ["llama3", "qwen2.5"])
    monkeypatch.setattr(
        "calute.tui.cli.launch_tui", lambda executor, agent, profile=None, config_store=None: _Launcher()
    )
    monkeypatch.setattr("calute.tui.cli.TerminalConfigStore", lambda: TerminalConfigStore(tmp_path / "profiles.json"))

    exit_code = main(["--provider", "ollama", "--model", "llama3", "--no-tools"])
    assert exit_code == 0
    assert captured["launched"] is True


def test_resolve_profile_heals_ollama_profile_pointing_at_v1_endpoint(tmp_path):
    store = TerminalConfigStore(tmp_path / "profiles.json")
    store.upsert_profile(
        TerminalProfile(
            name="default",
            provider="ollama",
            model="llama3",
            api_key="sk-xxx",
            base_url="http://35.193.63.250:11556/v1/",
            available_models=["llama3", "qwen2.5"],
        )
    )

    args = argparse.Namespace(
        provider=None,
        model=None,
        api_key=None,
        base_url=None,
        profile_name="default",
        list_profiles=False,
        list_models=False,
        choose_model=False,
        agent_id=None,
        instructions=None,
        profile=None,
        no_tools=False,
        command="tui",
    )

    profile = _resolve_profile(args, store, {})

    assert profile.provider == "openai"
    assert profile.base_url == "http://35.193.63.250:11556/v1/"
    assert profile.api_key == "sk-xxx"
    assert profile.available_models == []


def test_main_saves_profile(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class _Launcher:
        def launch(self):
            captured["launched"] = True

    monkeypatch.setattr("calute.tui.cli.create_llm", lambda provider, **kwargs: object())
    monkeypatch.setattr(
        "calute.tui.cli.discover_available_models",
        lambda *args, **kwargs: ["qwen3-coder", "deepseek-r1"],
    )
    monkeypatch.setattr(
        "calute.tui.cli.launch_tui", lambda executor, agent, profile=None, config_store=None: _Launcher()
    )
    monkeypatch.setattr("calute.tui.cli.TerminalConfigStore", lambda: TerminalConfigStore(tmp_path / "profiles.json"))

    exit_code = main(
        [
            "--provider",
            "openai",
            "--base-url",
            "http://35.193.63.250:11556/v1/",
            "--api-key",
            "sk-xxx",
            "--profile-name",
            "lab",
        ]
    )
    assert exit_code == 0
    assert captured["launched"] is True

    store = TerminalConfigStore(tmp_path / "profiles.json")
    saved = store.get_profile("lab")
    assert saved is not None
    assert saved.provider == "openai"
    assert saved.base_url == "http://35.193.63.250:11556/v1/"
    assert saved.api_key == "sk-xxx"
    assert saved.model == "qwen3-coder"
    assert saved.available_models == ["qwen3-coder", "deepseek-r1"]


def test_main_loads_power_tools_enabled_from_profile(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class _Launcher:
        def __init__(self, executor):
            self.executor = executor

        def launch(self):
            captured["power_tools_enabled"] = (
                self.executor._runtime_features_state.operator_state.config.power_tools_enabled
            )

    store = TerminalConfigStore(tmp_path / "profiles.json")
    store.upsert_profile(
        TerminalProfile(
            name="lab",
            provider="openai",
            model="qwen3-coder",
            api_key="sk-xxx",
            base_url="http://35.193.63.250:11556/v1/",
            power_tools_enabled=True,
        )
    )

    monkeypatch.setattr("calute.tui.cli.TerminalConfigStore", lambda: store)
    monkeypatch.setattr(
        "calute.tui.cli.create_llm",
        lambda provider, **kwargs: SimpleNamespace(config=SimpleNamespace(model=kwargs.get("model"))),
    )
    monkeypatch.setattr("calute.tui.cli.discover_available_models", lambda *args, **kwargs: ["qwen3-coder"])
    monkeypatch.setattr(
        "calute.tui.cli.launch_tui",
        lambda executor, agent, profile=None, config_store=None: _Launcher(executor),
    )

    exit_code = main(["--profile-name", "lab"])
    assert exit_code == 0
    assert captured["power_tools_enabled"] is True


def test_main_skips_model_discovery_when_profile_already_has_model(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class _Launcher:
        def launch(self):
            captured["launched"] = True

    store = TerminalConfigStore(tmp_path / "profiles.json")
    store.upsert_profile(
        TerminalProfile(
            name="lab",
            provider="openai",
            model="qwen3_5-27.36b",
            api_key="sk-xxx",
            base_url="http://34.68.24.241:11556/v1",
        )
    )

    monkeypatch.setattr("calute.tui.cli.TerminalConfigStore", lambda: store)
    monkeypatch.setattr(
        "calute.tui.cli.create_llm",
        lambda provider, **kwargs: SimpleNamespace(config=SimpleNamespace(model=kwargs.get("model"))),
    )

    def _fail_discovery(*args, **kwargs):
        raise AssertionError("discover_available_models should not run for normal startup when model is preset")

    monkeypatch.setattr("calute.tui.cli.discover_available_models", _fail_discovery)
    monkeypatch.setattr(
        "calute.tui.cli.launch_tui", lambda executor, agent, profile=None, config_store=None: _Launcher()
    )

    exit_code = main(["--profile-name", "lab"])
    assert exit_code == 0
    assert captured["launched"] is True


def test_resolve_profile_defaults_power_tools_enabled_for_new_profiles(tmp_path):
    store = TerminalConfigStore(tmp_path / "profiles.json")
    args = argparse.Namespace(
        provider="ollama",
        model="llama3",
        api_key=None,
        base_url=None,
        profile_name="default",
        list_profiles=False,
        list_models=False,
        choose_model=False,
        agent_id=None,
        instructions=None,
        profile=None,
        no_tools=False,
        command="tui",
    )

    profile = _resolve_profile(args, store, {})

    assert profile.power_tools_enabled is True
