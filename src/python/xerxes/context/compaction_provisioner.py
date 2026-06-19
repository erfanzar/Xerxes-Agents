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
"""Agent-backed context compaction and pre-append provisioning."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .token_counter import SmartTokenCounter

logger = logging.getLogger(__name__)

COMPACTION_SUMMARY_PREFIX = "[Previous conversation summary"
_DEFAULT_THRESHOLD_RATIO = 0.75
_DEFAULT_TARGET_RATIO = 0.5
_DEFAULT_TAIL_RATIO = 0.35
_LOCAL_PROVIDER_NAMES = {"custom", "lmstudio", "ollama"}

CompactionSummaryAgent = Callable[[list[dict[str, Any]], str | None], str]


@dataclass(frozen=True)
class CompactionProvision:
    """Result from one compaction provisioning pass."""

    compacted: bool
    messages: list[dict[str, Any]]
    tokens_before: int
    tokens_after: int
    summarized_count: int = 0
    kept_count: int = 0
    reason: str = ""
    error: str = ""


def message_content_to_text(content: Any) -> str:
    """Return a readable text rendering for provider message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or json.dumps(item, ensure_ascii=False, default=str)))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, default=str)
    return str(content or "")


def render_messages_for_summary(messages: list[dict[str, Any]]) -> str:
    """Render messages as complete compaction-agent input."""
    rendered = []
    for index, message in enumerate(messages, start=1):
        role = str(message.get("role", "unknown")).upper()
        lines = [f"Message {index} [{role}]"]
        content = message_content_to_text(message.get("content", ""))
        if content:
            lines.append(content)
        if message.get("tool_calls"):
            lines.append("tool_calls=" + json.dumps(message.get("tool_calls"), ensure_ascii=False, default=str))
        if message.get("tool_call_id"):
            lines.append(f"tool_call_id={message.get('tool_call_id')}")
        rendered.append("\n".join(lines))
    return "\n\n".join(rendered)


def _tool_call_names(message: dict[str, Any]) -> dict[str, str]:
    """Return ``tool_call_id -> tool name`` for an assistant message."""
    names: dict[str, str] = {}
    for tool_call in message.get("tool_calls") or []:
        tool_call_id = tool_call.get("id")
        if tool_call_id:
            names[str(tool_call_id)] = str(tool_call.get("name") or "")
    return names


def _complete_pending_tool_results(out: list[dict[str, Any]], pending: dict[str, str]) -> None:
    """Append synthetic tool results for assistant calls that lost their result."""
    for tool_call_id, name in pending.items():
        out.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": "[Tool result unavailable after context compaction]",
                "is_error": True,
            }
        )
    pending.clear()


def repair_tool_message_sequence(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return messages with provider-valid assistant/tool-call ordering.

    OpenAI-compatible providers reject a ``tool`` message unless it answers
    a preceding assistant ``tool_calls`` entry. Compaction changes the leading
    edge of the request window, so orphan tool results are dropped and missing
    recent tool results are represented by synthetic error messages.
    """
    out: list[dict[str, Any]] = []
    pending: dict[str, str] = {}

    for message in messages:
        role = message.get("role")
        if role == "assistant":
            if pending:
                _complete_pending_tool_results(out, pending)
            out.append(message)
            pending = _tool_call_names(message)
            continue

        if role == "tool":
            tool_call_id = str(message.get("tool_call_id") or "")
            if tool_call_id and tool_call_id in pending:
                out.append(message)
                pending.pop(tool_call_id, None)
            continue

        if pending:
            _complete_pending_tool_results(out, pending)
        out.append(message)

    if pending:
        _complete_pending_tool_results(out, pending)
    return out


class ProviderCompactionAgent:
    """LLM-backed summary agent used when no test or caller agent is injected."""

    def __init__(self, *, model: str, config: dict[str, Any], max_tokens: int = 8192) -> None:
        """Bind provider config for summary calls."""
        self.model = model
        self.config = config
        self.max_tokens = max(512, int(max_tokens))

    def __call__(self, messages: list[dict[str, Any]], previous_summary: str | None = None) -> str:
        """Ask the configured provider to rewrite compactable messages."""
        from ..llms.registry import bare_model, detect_provider, get_provider_config

        provider_name = detect_provider(self.model)
        provider_config = get_provider_config(provider_name)
        model_name = bare_model(self.model)
        prompt = self._build_prompt(messages, previous_summary)

        if provider_config.type == "anthropic":
            return self._call_anthropic(provider_name, model_name, prompt)
        return self._call_openai_compatible(provider_name, model_name, prompt)

    @staticmethod
    def _build_prompt(messages: list[dict[str, Any]], previous_summary: str | None) -> str:
        """Build the compaction prompt sent to the provider."""
        prior = f"Existing summary to refresh:\n{previous_summary.strip()}\n\n" if previous_summary else ""
        return (
            "Rewrite the following conversation history into a compact, durable memory for the next model turn.\n"
            "Preserve concrete facts, user instructions, decisions, file paths, tool results, errors, fixes, "
            "open questions, and current task state. Drop chatter and duplicate text. Output only the summary.\n\n"
            f"{prior}"
            f"Conversation history to compact:\n{render_messages_for_summary(messages)}"
        )

    def _request_timeout(self) -> float:
        """Return provider request timeout for summary calls."""
        raw = self.config.get("llm_timeout") or self.config.get("request_timeout")
        try:
            return max(1.0, float(raw)) if raw is not None else 60.0
        except (TypeError, ValueError):
            return 60.0

    def _call_openai_compatible(self, provider_name: str, model_name: str, prompt: str) -> str:
        """Run a summary call against an OpenAI-compatible provider."""
        from openai import OpenAI

        from ..llms.registry import PROVIDERS, get_api_key, provider_default_headers

        provider = PROVIDERS.get(provider_name, PROVIDERS.get("openai"))
        api_key = self.config.get("api_key") or get_api_key(provider_name, self.config) or "dummy"
        base_url = (
            self.config.get("base_url")
            or self.config.get("custom_base_url")
            or (provider.base_url if provider else None)
            or "https://api.openai.com/v1"
        )
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=provider_default_headers(provider_name) or None,
            timeout=self._request_timeout(),
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a context compaction agent. Output only the rewritten summary.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    def _call_anthropic(self, provider_name: str, model_name: str, prompt: str) -> str:
        """Run a summary call against Anthropic Messages."""
        import anthropic

        from ..llms.registry import get_api_key

        client = anthropic.Anthropic(
            api_key=self.config.get("api_key") or get_api_key(provider_name, self.config),
            timeout=self._request_timeout(),
        )
        response = client.messages.create(
            model=model_name,
            system="You are a context compaction agent. Output only the rewritten summary.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        parts = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)


def compaction_summary_agent_from_config(
    model: str,
    config: dict[str, Any],
) -> CompactionSummaryAgent | None:
    """Return an injected or provider-backed compaction summary agent."""
    injected = config.get("compaction_summary_agent")
    if callable(injected):
        return injected
    if injected is False:
        return None
    if not model:
        return None

    from ..llms.registry import PROVIDERS, detect_provider, get_api_key

    provider_name = detect_provider(model)
    provider = PROVIDERS.get(provider_name)
    explicit_base_url = bool(config.get("base_url") or config.get("custom_base_url"))
    api_key = config.get("api_key") or get_api_key(provider_name, config)

    if not api_key and provider_name not in _LOCAL_PROVIDER_NAMES and not explicit_base_url:
        return None
    if provider and provider.type == "anthropic" and not api_key:
        return None

    return ProviderCompactionAgent(
        model=model,
        config=config,
        max_tokens=int(config.get("compaction_summary_max_tokens") or 8192),
    )


class CompactionProvisioner:
    """Plan and execute agent-backed context compaction."""

    def __init__(
        self,
        *,
        model: str,
        max_context_tokens: int,
        summary_agent: CompactionSummaryAgent | None,
        threshold_tokens: int | None = None,
        target_tokens: int | None = None,
        threshold_ratio: float = _DEFAULT_THRESHOLD_RATIO,
        target_ratio: float = _DEFAULT_TARGET_RATIO,
        tail_ratio: float = _DEFAULT_TAIL_RATIO,
    ) -> None:
        """Configure token budgets and the summary agent."""
        self.model = model
        self.max_context_tokens = max(1, int(max_context_tokens or 1))
        self.threshold_tokens = threshold_tokens or int(self.max_context_tokens * threshold_ratio)
        self.target_tokens = target_tokens or int(self.max_context_tokens * target_ratio)
        self.threshold_tokens = max(1, self.threshold_tokens)
        self.target_tokens = max(1, min(self.target_tokens, self.threshold_tokens))
        self.tail_ratio = max(0.05, min(0.9, tail_ratio))
        self.summary_agent = summary_agent
        self.token_counter = SmartTokenCounter(model=model)

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Return estimated tokens for ``messages``."""
        return self.token_counter.count_tokens(messages)

    def should_compact(
        self,
        messages: list[dict[str, Any]],
        *,
        appended_messages: list[dict[str, Any]] | None = None,
        force: bool = False,
    ) -> bool:
        """Return whether a compaction pass should run now."""
        if force:
            return True
        candidate = [*messages, *(appended_messages or [])]
        return self.count_tokens(candidate) >= self.threshold_tokens

    def compact_before_append(
        self,
        messages: list[dict[str, Any]],
        appended_messages: list[dict[str, Any]],
    ) -> CompactionProvision:
        """Compact existing context when appending new messages would overflow."""
        candidate = [*messages, *appended_messages]
        tokens_before = self.count_tokens(candidate)
        if tokens_before < self.threshold_tokens:
            return CompactionProvision(False, messages, tokens_before, tokens_before, reason="below_threshold")

        provision = self.compact(messages, force=True)
        if not provision.compacted:
            return CompactionProvision(
                False,
                messages,
                tokens_before,
                tokens_before,
                reason=provision.reason,
                error=provision.error,
            )
        return CompactionProvision(
            True,
            provision.messages,
            tokens_before,
            self.count_tokens([*provision.messages, *appended_messages]),
            summarized_count=provision.summarized_count,
            kept_count=provision.kept_count,
            reason=provision.reason,
        )

    def compact(
        self,
        messages: list[dict[str, Any]],
        *,
        force: bool = False,
        previous_summary: str | None = None,
    ) -> CompactionProvision:
        """Replace compactable history with an agent-written summary."""
        tokens_before = self.count_tokens(messages)
        if not force and tokens_before < self.threshold_tokens:
            return CompactionProvision(False, messages, tokens_before, tokens_before, reason="below_threshold")

        if self.summary_agent is None:
            return CompactionProvision(False, messages, tokens_before, tokens_before, reason="no_summary_agent")

        partition = self._partition(messages)
        if partition is None:
            return CompactionProvision(False, messages, tokens_before, tokens_before, reason="nothing_to_compact")

        system_messages, compactable_messages, live_messages = partition
        try:
            summary = self.summary_agent(compactable_messages, previous_summary).strip()
        except Exception as exc:
            logger.warning("Context compaction agent failed", exc_info=True)
            return CompactionProvision(
                False,
                messages,
                tokens_before,
                tokens_before,
                reason="summary_agent_failed",
                error=str(exc),
            )

        if not summary:
            return CompactionProvision(False, messages, tokens_before, tokens_before, reason="empty_summary")

        summary_message = {
            "role": "user",
            "content": f"{COMPACTION_SUMMARY_PREFIX} - {len(compactable_messages)} messages compacted]\n\n{summary}",
        }
        compacted = repair_tool_message_sequence([*system_messages, summary_message, *live_messages])
        tokens_after = self.count_tokens(compacted)
        if tokens_after >= tokens_before:
            return CompactionProvision(False, messages, tokens_before, tokens_after, reason="summary_did_not_shrink")

        return CompactionProvision(
            True,
            compacted,
            tokens_before,
            tokens_after,
            summarized_count=len(compactable_messages),
            kept_count=len(live_messages),
            reason="compacted",
        )

    def _partition(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
        """Split messages into system, compactable, and live suffix by token budget."""
        system_messages = [message for message in messages if message.get("role") == "system"]
        conversation_messages = [message for message in messages if message.get("role") != "system"]
        if len(conversation_messages) < 2:
            return None

        tail_budget = max(1, int(self.target_tokens * self.tail_ratio))
        live_messages: list[dict[str, Any]] = []
        live_tokens = 0
        for message in reversed(conversation_messages):
            message_tokens = max(1, self.count_tokens([message]))
            if live_messages and live_tokens + message_tokens > tail_budget:
                break
            live_messages.insert(0, message)
            live_tokens += message_tokens

        if not live_messages:
            live_messages = [conversation_messages[-1]]

        start = len(conversation_messages) - len(live_messages)
        while start > 0 and conversation_messages[start].get("role") == "tool":
            start -= 1
            if conversation_messages[start].get("role") == "assistant":
                break

        compactable_messages = conversation_messages[:start]
        live_messages = conversation_messages[start:]
        if not compactable_messages or not live_messages:
            return None
        return system_messages, compactable_messages, live_messages
