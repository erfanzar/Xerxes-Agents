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
"""Verify the status bar's ``max_context`` falls back to the model's published window.

Previously the bar showed ``0/0`` whenever the user hadn't explicitly set
``context_limit`` in runtime config — visually broken on every fresh launch.
``_resolve_context_limit`` now consults
:func:`xerxes.llms.registry.get_context_limit` as the fallback, so the
denominator always reflects something sensible.
"""

from __future__ import annotations

from types import SimpleNamespace

from xerxes.context.window_usage import estimate_context_tokens
from xerxes.daemon.config import DaemonConfig
from xerxes.daemon.runtime import RuntimeManager, TurnRunner
from xerxes.streaming.events import AgentState


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
    # Kimi-code provider's published window is 256k.
    assert runner._resolve_context_limit() == 256_000


def test_falls_back_to_anthropic_window(tmp_path):
    runner = _make_runner(tmp_path, {"model": "claude-sonnet-4-6"})
    assert runner._resolve_context_limit() == 1_000_000


def test_returns_zero_when_no_model(tmp_path):
    runner = _make_runner(tmp_path, {})
    assert runner._resolve_context_limit() == 0


def test_status_payload_reports_live_context_not_cumulative_usage(tmp_path):
    runner = _make_runner(tmp_path, {"model": "gpt-4o", "context_limit": 100_000})
    session = SimpleNamespace(
        state=AgentState(
            messages=[{"role": "user", "content": "short prompt"}],
            total_input_tokens=554_000,
            total_output_tokens=582,
        )
    )

    payload = runner._status_payload(session, mode="plan", plan_mode=True)

    assert payload["context_tokens"] < 100
    assert payload["max_context"] == 100_000


def test_status_payload_counts_system_prompt_and_tool_schemas(tmp_path):
    runner = _make_runner(tmp_path, {"model": "gpt-4o", "context_limit": 100_000})
    runner.runtime.system_prompt = "system prompt " * 100
    runner.runtime.tool_schemas = [
        {
            "name": "BigTool",
            "description": "large schema " * 500,
            "input_schema": {"type": "object", "properties": {"value": {"type": "string"}}},
        }
    ]
    session = SimpleNamespace(
        state=AgentState(messages=[{"role": "user", "content": "short prompt"}]),
        runtime_config={"model": "gpt-4o", "context_limit": 100_000},
    )

    payload = runner._status_payload(session, mode="code", plan_mode=False)
    message_only = estimate_context_tokens(session.state.messages, model="gpt-4o")

    assert payload["context_tokens"] > message_only
    assert payload["max_context"] == 100_000
