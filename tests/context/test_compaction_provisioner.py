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
"""Agent-backed compaction provisioner tests."""

from __future__ import annotations

from typing import Any

from xerxes.context.compaction_provisioner import (
    CompactionProvisioner,
    ProviderCompactionAgent,
    repair_tool_message_sequence,
)


def _messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "old request " * 100},
        {"role": "assistant", "content": "old answer " * 100},
        {"role": "user", "content": "live request"},
    ]


def test_compaction_uses_summary_agent_without_local_summary() -> None:
    calls: list[list[dict[str, Any]]] = []

    def summary_agent(messages: list[dict[str, Any]], _previous_summary: str | None) -> str:
        calls.append(messages)
        return "AGENT MEMORY"

    provisioner = CompactionProvisioner(
        model="gpt-4o",
        max_context_tokens=200,
        threshold_tokens=1,
        target_tokens=40,
        summary_agent=summary_agent,
    )

    result = provisioner.compact(_messages(), force=True)

    assert result.compacted is True
    assert calls
    summary = next(
        message["content"] for message in result.messages if "Previous conversation summary" in message["content"]
    )
    assert "AGENT MEMORY" in summary
    assert "old request" not in summary


def test_compaction_requires_summary_agent() -> None:
    provisioner = CompactionProvisioner(
        model="gpt-4o",
        max_context_tokens=200,
        threshold_tokens=1,
        target_tokens=40,
        summary_agent=None,
    )

    result = provisioner.compact(_messages(), force=True)

    assert result.compacted is False
    assert result.reason == "no_summary_agent"


def test_compact_before_append_compacts_existing_messages_only() -> None:
    def summary_agent(messages: list[dict[str, Any]], _previous_summary: str | None) -> str:
        return f"summary for {len(messages)} messages"

    provisioner = CompactionProvisioner(
        model="gpt-4o",
        max_context_tokens=200,
        threshold_tokens=1,
        target_tokens=40,
        summary_agent=summary_agent,
    )
    incoming = [{"role": "user", "content": "new turn " * 100}]

    result = provisioner.compact_before_append(_messages(), incoming)

    assert result.compacted is True
    assert all(message.get("content") != incoming[0]["content"] for message in result.messages)
    assert "summary for" in result.messages[1]["content"]


def test_compact_before_append_without_agent_does_not_drop_history() -> None:
    provisioner = CompactionProvisioner(
        model="gpt-4o",
        max_context_tokens=200,
        threshold_tokens=1,
        target_tokens=40,
        summary_agent=None,
    )
    messages = _messages()
    incoming = [{"role": "user", "content": "new turn " * 100}]

    result = provisioner.compact_before_append(messages, incoming)

    assert result.compacted is False
    assert result.messages == messages
    assert result.reason == "no_summary_agent"


def test_repair_tool_message_sequence_drops_orphans_and_backfills_missing_results() -> None:
    messages = [
        {"role": "tool", "tool_call_id": "orphan", "name": "X", "content": "drop"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "missing", "name": "Y", "input": {}}],
        },
        {"role": "user", "content": "continue"},
    ]

    repaired = repair_tool_message_sequence(messages)

    assert all(message.get("tool_call_id") != "orphan" for message in repaired)
    tool_messages = [message for message in repaired if message.get("role") == "tool"]
    assert tool_messages[0]["tool_call_id"] == "missing"
    assert tool_messages[0]["is_error"] is True


def test_provider_compaction_agent_resolves_kimi_code_saved_profile(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_call_openai_compatible(
        self: ProviderCompactionAgent,
        provider_name: str,
        model_name: str,
        prompt: str,
    ) -> str:
        captured["provider_name"] = provider_name
        captured["model_name"] = model_name
        captured["prompt"] = prompt
        return "summary"

    monkeypatch.setattr(ProviderCompactionAgent, "_call_openai_compatible", fake_call_openai_compatible)

    agent = ProviderCompactionAgent(
        model="kimi/kimi-for-coding",
        config={"base_url": "https://api.kimi.com/coding/v1"},
    )

    assert agent([{"role": "user", "content": "old context"}]) == "summary"
    assert captured["provider_name"] == "kimi-code"
    assert captured["model_name"] == "kimi-for-coding"
    assert "old context" in captured["prompt"]
