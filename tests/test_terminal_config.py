# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from pathlib import Path

from calute.tui.terminal_config import TerminalConfigStore, TerminalProfile


def test_terminal_config_round_trip(tmp_path: Path):
    store = TerminalConfigStore(tmp_path / "terminal_profiles.json")
    profile = TerminalProfile(
        name="work",
        provider="openai",
        model="qwen3-coder",
        api_key="sk-test",
        base_url="http://0.0.0.0:11556/v1/",
        power_tools_enabled=True,
        available_models=["qwen3-coder", "deepseek-v3"],
        sampling_params={"temperature": 0.3, "max_tokens": 4096},
    )

    store.upsert_profile(profile)
    loaded = store.get_profile("work")

    assert loaded is not None
    assert loaded.provider == "openai"
    assert loaded.model == "qwen3-coder"
    assert loaded.base_url == "http://0.0.0.0:11556/v1/"
    assert loaded.power_tools_enabled is True
    assert loaded.available_models == ["qwen3-coder", "deepseek-v3"]
    assert loaded.sampling_params == {"temperature": 0.3, "max_tokens": 4096}


def test_terminal_config_uses_last_profile(tmp_path: Path):
    store = TerminalConfigStore(tmp_path / "terminal_profiles.json")
    store.upsert_profile(TerminalProfile(name="default", provider="ollama", model="llama3"))
    store.upsert_profile(TerminalProfile(name="lab", provider="openai", model="gpt-4o"), make_default=True)

    loaded = store.get_profile()
    assert loaded is not None
    assert loaded.name == "lab"


def test_terminal_config_normalizes_invalid_prompt_profile(tmp_path: Path):
    path = tmp_path / "terminal_profiles.json"
    path.write_text(
        """
        {
          "last_profile": "default",
          "profiles": {
            "default": {
              "name": "default",
              "provider": "openai",
              "model": "qwen3",
              "prompt_profile": "weird"
            }
          }
        }
        """,
        encoding="utf-8",
    )

    store = TerminalConfigStore(path)
    loaded = store.get_profile()
    assert loaded is not None
    assert loaded.prompt_profile == "full"


def test_terminal_config_missing_power_tools_defaults_true(tmp_path: Path):
    path = tmp_path / "terminal_profiles.json"
    path.write_text(
        """
        {
          "last_profile": "default",
          "profiles": {
            "default": {
              "name": "default",
              "provider": "openai",
              "model": "qwen3"
            }
          }
        }
        """,
        encoding="utf-8",
    )

    store = TerminalConfigStore(path)
    loaded = store.get_profile()
    assert loaded is not None
    assert loaded.power_tools_enabled is True
