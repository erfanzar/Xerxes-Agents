# Copyright 2026 Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
"""Verify the status bar's ``max_context`` falls back to the model's published window.

Previously the bar showed ``0/0`` whenever the user hadn't explicitly set
``context_limit`` in runtime config — visually broken on every fresh launch.
``_resolve_context_limit`` now consults
:func:`xerxes.llms.registry.get_context_limit` as the fallback, so the
denominator always reflects something sensible.
"""

from __future__ import annotations

from xerxes.daemon.config import DaemonConfig
from xerxes.daemon.runtime import RuntimeManager, TurnRunner


def _make_runner(tmp_path, runtime_overrides: dict) -> TurnRunner:
    runtime = RuntimeManager(DaemonConfig(project_dir=str(tmp_path)))
    runtime.runtime_config = runtime_overrides
    runner = TurnRunner.__new__(TurnRunner)
    runner.runtime = runtime
    return runner


def test_explicit_context_limit_wins(tmp_path):
    runner = _make_runner(tmp_path, {"model": "kimi-for-coding", "context_limit": 99_999})
    assert runner._resolve_context_limit() == 99_999


def test_max_context_alias_honoured(tmp_path):
    runner = _make_runner(tmp_path, {"model": "kimi-for-coding", "max_context": 50_000})
    assert runner._resolve_context_limit() == 50_000


def test_falls_back_to_registry_for_known_provider(tmp_path):
    runner = _make_runner(tmp_path, {"model": "kimi-for-coding"})
    # Kimi-code provider's published window is 128k.
    assert runner._resolve_context_limit() == 128_000


def test_falls_back_to_anthropic_window(tmp_path):
    runner = _make_runner(tmp_path, {"model": "claude-sonnet-4-6"})
    # Anthropic per-model registry returns 200k.
    assert runner._resolve_context_limit() == 200_000


def test_returns_zero_when_no_model(tmp_path):
    runner = _make_runner(tmp_path, {})
    assert runner._resolve_context_limit() == 0
