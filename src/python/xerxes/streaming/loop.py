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
"""Core agent loop: stream tokens, route tool calls, and gate on permissions.

The loop is the single sync generator that drives a turn end-to-end: it
appends the user message, streams from the active provider (Anthropic or any
OpenAI-compatible endpoint), parses thinking tags, dispatches tool calls
through the permission system, executes the tools, and feeds results back
until the model stops requesting tools or an explicit ``max_tool_turns`` budget is exhausted.

:func:`arun` adapts the sync generator to an async generator by running
``next()`` in the default executor. ``_stream_anthropic`` and
``_stream_openai_compat`` are the provider adapters; the rest are private
helpers for thinking-tag splitting, prompt caching, and request-config
parsing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Generator
from pathlib import Path
from typing import Any

from ..context.compaction_provisioner import (
    CompactionProvisioner,
    compaction_summary_agent_from_config,
)
from ..context.headroom import DEFAULT_HEADROOM_PREVIEW_CHARS, compress_tool_result
from ..context.window_usage import estimate_request_overhead_tokens
from ..llms.registry import get_context_limit
from ..runtime.change_guard import analyze_workspace_changes, format_change_guard_notification
from ..runtime.iteration_budget import iteration_budget_from_config
from ..runtime.objective_guard import inspect_objective_response, objective_guard_retry_limit
from ..runtime.workflow_memory import capture_user_workflow_memory
from .events import (
    AgentState,
    PermissionRequest,
    ProviderRetry,
    SkillSuggestion,
    StreamEvent,
    TextChunk,
    ThinkingChunk,
    ToolEnd,
    ToolStart,
    TurnDone,
)
from .messages import messages_to_anthropic, messages_to_openai
from .permissions import (
    PermissionMode,
    check_permission,
    format_permission_description,
)
from .prompt_caching import (
    extract_cache_tokens,
    wrap_system_with_cache,
    wrap_tools_with_cache,
)
from .tool_markers import extract_assistant_tool_call_markers


class _ThinkingParser:
    """Incremental splitter for ``<think>...</think>`` / ``<thinking>...</thinking>`` tags.

    Streamed text deltas may arrive with tags split across chunk boundaries;
    the parser buffers the partial input, emits :class:`TextChunk` for visible
    text and :class:`ThinkingChunk` for reasoning, and flushes any trailing
    thinking buffer when fed an empty chunk.

    Attributes:
        _OPEN_TAGS: Recognised opening tags.
        _CLOSE_TAGS: Recognised closing tags.
    """

    __slots__ = ("_buffer", "_in_thinking", "_thinking_buf")

    _OPEN_TAGS: tuple[str, ...] = ("<think>", "<thinking>")
    _CLOSE_TAGS: tuple[str, ...] = ("</think>", "</thinking>")

    def __init__(self) -> None:
        """Initialise empty buffers and start in visible-text mode."""
        self._buffer = ""
        self._in_thinking = False
        self._thinking_buf = ""

    @staticmethod
    def _find_any(text: str, tags: tuple[str, ...]) -> tuple[int, str]:
        """Return the earliest occurrence of any tag and the tag itself.

        Returns ``(-1, "")`` when no tag matches.
        """

        earliest_idx = -1
        earliest_tag = ""
        for tag in tags:
            idx = text.find(tag)
            if idx != -1 and (earliest_idx == -1 or idx < earliest_idx):
                earliest_idx = idx
                earliest_tag = tag
        return earliest_idx, earliest_tag

    @staticmethod
    def _partial_tail(text: str, tags: tuple[str, ...]) -> int:
        """Length of the longest trailing run of ``text`` that is a proper prefix of a tag.

        Returns ``0`` when no non-empty suffix of ``text`` starts any tag. The
        held-back suffix is never the full tag (a complete tag is matched by
        :meth:`_find_any` instead), so callers must keep it buffered until the
        next chunk arrives in case the tag straddles a chunk boundary.
        """

        max_len = 0
        for tag in tags:
            # A split tag can hold back at most len(tag) - 1 characters.
            limit = min(len(text), len(tag) - 1)
            for k in range(limit, 0, -1):
                if text[-k:] == tag[:k]:
                    if k > max_len:
                        max_len = k
                    break
        return max_len

    def process(self, chunk_text: str) -> list[TextChunk | ThinkingChunk]:
        """Feed a streamed text fragment and emit any completed chunks.

        Pass ``""`` after the stream ends to flush a still-open thinking
        block. The parser preserves state across calls, so partial tags
        straddling chunk boundaries are handled correctly.

        Args:
            chunk_text: Newly received text delta (may be empty for flush).

        Returns:
            Ordered list of visible-text and reasoning chunks ready to yield
            downstream.
        """

        events: list[TextChunk | ThinkingChunk] = []
        self._buffer += chunk_text
        final_flush = not chunk_text

        if final_flush and self._in_thinking:
            # End of stream while still inside a thinking block: any buffered
            # partial close tag we held back can never complete, so treat it as
            # reasoning content rather than dropping it.
            if self._buffer:
                self._thinking_buf += self._buffer
                self._buffer = ""
            if self._thinking_buf:
                events.append(ThinkingChunk(self._thinking_buf))
            self._thinking_buf = ""
            self._in_thinking = False
            return events

        while self._buffer:
            if not self._in_thinking:
                idx, tag = self._find_any(self._buffer, self._OPEN_TAGS)
                if idx == -1:
                    # No complete open tag. Hold back a trailing fragment that
                    # could be the start of one split across the next chunk
                    # (e.g. "<thi" before "nk>"). On the final flush nothing
                    # more is coming, so emit the whole buffer verbatim.
                    hold = 0 if final_flush else self._partial_tail(self._buffer, self._OPEN_TAGS)
                    emit = self._buffer[: len(self._buffer) - hold] if hold else self._buffer
                    if emit:
                        events.append(TextChunk(emit))
                    self._buffer = self._buffer[len(self._buffer) - hold :] if hold else ""
                    break
                if idx > 0:
                    events.append(TextChunk(self._buffer[:idx]))
                self._buffer = self._buffer[idx + len(tag) :]
                self._in_thinking = True
                self._thinking_buf = ""

            else:
                idx, tag = self._find_any(self._buffer, self._CLOSE_TAGS)
                if idx == -1:
                    # No complete close tag. Hold back a trailing fragment that
                    # could be the start of one split across the next chunk
                    # (e.g. "</thi" before "nk>") so the visible answer after it
                    # is not swallowed into the thinking channel.
                    hold = 0 if final_flush else self._partial_tail(self._buffer, self._CLOSE_TAGS)
                    keep = self._buffer[: len(self._buffer) - hold] if hold else self._buffer
                    self._thinking_buf += keep
                    self._buffer = self._buffer[len(self._buffer) - hold :] if hold else ""
                    break
                if idx > 0:
                    self._thinking_buf += self._buffer[:idx]
                self._buffer = self._buffer[idx + len(tag) :]
                self._in_thinking = False
                if self._thinking_buf:
                    events.append(ThinkingChunk(self._thinking_buf))
                    self._thinking_buf = ""

        return events


def _parse_thinking_tags(
    text: str,
) -> tuple[str, str]:
    """Split a complete text blob into ``(visible, thinking)``.

    Convenience over :class:`_ThinkingParser` for non-streaming callers.
    """

    thinking_pattern = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL)
    thinking_parts = thinking_pattern.findall(text)
    thinking_text = "".join(thinking_parts).strip()
    visible_text = thinking_pattern.sub("", text).strip()
    return visible_text, thinking_text


logger = logging.getLogger(__name__)

_RETRYABLE_STREAM_EXCEPTIONS: tuple[type[Exception], ...] = (ConnectionError, TimeoutError)
try:
    import openai

    _RETRYABLE_STREAM_EXCEPTIONS += (openai.APIError,)
except Exception:
    pass
try:
    import httpx

    _RETRYABLE_STREAM_EXCEPTIONS += (httpx.HTTPError,)
except Exception:
    pass

_DEFAULT_CONTEXT_LIMIT = 128_000
_DEFAULT_CONTEXT_SAFETY_TOKENS = 4_096
_REQUEST_OVERHEAD_TOKENS_CONFIG_KEY = "_request_overhead_tokens"
_CONTEXT_LIMIT_ERROR_MARKERS = (
    "exceeded model token limit",
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "too many tokens",
)
_CONTEXT_REQUESTED_RE = re.compile(r"\brequested\D+(\d{4,})", re.IGNORECASE)
_CONTEXT_LIMIT_RE = re.compile(r"\b(?:model token limit|token limit|context limit|limit)\D+(\d{4,})", re.IGNORECASE)


def _is_context_limit_error(exc: Exception) -> bool:
    """Return true for provider errors that cannot succeed by retrying unchanged."""
    text = str(exc).lower()
    return any(marker in text for marker in _CONTEXT_LIMIT_ERROR_MARKERS)


def _context_limit_numbers(exc: Exception) -> tuple[int | None, int | None]:
    """Extract provider-reported ``requested`` and ``limit`` token counts."""
    text = str(exc)
    requested_match = _CONTEXT_REQUESTED_RE.search(text)
    limit_match = _CONTEXT_LIMIT_RE.search(text)
    requested = int(requested_match.group(1)) if requested_match else None
    limit = int(limit_match.group(1)) if limit_match else None
    return requested, limit


def _calibrate_request_overhead_from_context_error(
    state: AgentState,
    *,
    config: dict[str, Any],
    model: str,
    error: Exception,
) -> dict[str, int]:
    """Raise request-overhead accounting when the provider reports a larger window."""
    requested, provider_limit = _context_limit_numbers(error)
    if requested is None:
        return {}

    provisioner = _compaction_provisioner(config=config, model=model)
    message_tokens = provisioner.count_tokens(state.messages)
    observed_overhead = max(0, requested - message_tokens)
    current_overhead = _request_overhead_tokens(config)
    if observed_overhead > current_overhead:
        config[_REQUEST_OVERHEAD_TOKENS_CONFIG_KEY] = observed_overhead

    payload = {
        "provider_requested_tokens": requested,
        "estimated_message_tokens": message_tokens,
        "request_overhead_tokens": _request_overhead_tokens(config),
    }
    if provider_limit is not None:
        payload["provider_limit_tokens"] = provider_limit
    return payload


def _short_context_error_detail(value: Any, *, max_chars: int = 240) -> str:
    """Return a compact, single-line context failure detail."""
    text = str(value or "").strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _format_context_limit_stop(
    *,
    provision: dict[str, Any],
    observed: dict[str, int],
    reason: str | None = None,
) -> str:
    """Render a provider context-limit stop without echoing the raw provider error."""
    failure_reason = reason or str(provision.get("reason") or "compaction_unavailable")
    detail = _short_context_error_detail(provision.get("error"))
    suffix = f" ({detail})" if detail else ""
    requested = observed.get("provider_requested_tokens")
    provider_limit = observed.get("provider_limit_tokens") or provision.get("max_context_tokens")
    requested_text = ""
    if requested is not None and provider_limit is not None:
        requested_text = f" Provider reported {requested:,}/{int(provider_limit):,} tokens."
    return (
        "\n[Stopped: provider rejected the request as too large."
        f"{requested_text} Forced compaction could not reduce the request window: {failure_reason}{suffix}.]"
    )


def _context_safety_tokens(config: dict[str, Any], *, max_context_tokens: int | None = None) -> int:
    """Return the reserved token headroom before the provider hard limit."""
    explicit = "context_safety_tokens" in config or "context_reserve_tokens" in config
    raw = config.get("context_safety_tokens", config.get("context_reserve_tokens", _DEFAULT_CONTEXT_SAFETY_TOKENS))
    try:
        reserve = max(0, int(raw))
    except (TypeError, ValueError):
        reserve = _DEFAULT_CONTEXT_SAFETY_TOKENS
    if max_context_tokens is None:
        return reserve
    max_reserve = max(0, max_context_tokens - 1)
    if not explicit:
        max_reserve = min(max_reserve, max(0, int(max_context_tokens * 0.1)))
    return min(reserve, max_reserve)


def _request_overhead_tokens(config: dict[str, Any]) -> int:
    """Return tokens for request scaffolding outside ``state.messages``."""
    try:
        return max(0, int(config.get(_REQUEST_OVERHEAD_TOKENS_CONFIG_KEY) or 0))
    except (TypeError, ValueError):
        return 0


def _effective_context_limit(provisioner: CompactionProvisioner, config: dict[str, Any]) -> int:
    """Return the request limit after reserving tokenizer/provider slack."""
    return max(
        1,
        provisioner.max_context_tokens
        - _context_safety_tokens(config, max_context_tokens=provisioner.max_context_tokens),
    )


def _compaction_provisioner(
    *,
    config: dict[str, Any],
    model: str,
    budget_limit: int | None = None,
) -> CompactionProvisioner:
    """Build the shared compaction provisioner for the streaming loop."""
    context_limit = int(
        config.get("max_context_tokens") or get_context_limit(model) or budget_limit or _DEFAULT_CONTEXT_LIMIT
    )
    threshold_tokens = config.get("compaction_threshold_tokens")
    if threshold_tokens is None and budget_limit is not None:
        threshold_tokens = min(int(budget_limit), int(context_limit * float(config.get("compaction_threshold", 0.75))))
    target_tokens = config.get("compaction_target_tokens")
    summary_agent = compaction_summary_agent_from_config(model, config)
    return CompactionProvisioner(
        model=model,
        max_context_tokens=context_limit,
        threshold_tokens=int(threshold_tokens) if threshold_tokens is not None else None,
        target_tokens=int(target_tokens) if target_tokens is not None else None,
        threshold_ratio=float(config.get("compaction_threshold", 0.75)),
        target_ratio=float(config.get("compaction_target", 0.5)),
        summary_agent=summary_agent,
    )


def _try_compact_messages(
    state: AgentState,
    budget_limit: int,
    config: dict[str, Any] | None = None,
    model: str | None = None,
    *,
    force: bool = True,
) -> bool:
    """Try to compact messages using an agent-written summary."""
    cfg = config or {}
    selected_model = model or str(cfg.get("model", ""))
    provisioner = _compaction_provisioner(config=cfg, model=selected_model, budget_limit=budget_limit)
    provision = provisioner.compact(state.messages, force=force)
    if not provision.compacted:
        return False

    state.messages = provision.messages
    state.total_input_tokens = 0
    state.total_output_tokens = 0
    state.metadata["last_compaction"] = {
        "tokens_before": provision.tokens_before,
        "tokens_after": provision.tokens_after,
        "summarized_count": provision.summarized_count,
        "kept_count": provision.kept_count,
    }
    return True


def _compact_before_append(
    state: AgentState,
    messages: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    model: str,
) -> bool:
    """Compact existing context before appending ``messages`` when needed."""
    provisioner = _compaction_provisioner(config=config, model=model)
    overhead_tokens = _request_overhead_tokens(config)
    tokens_before = overhead_tokens + provisioner.count_tokens([*state.messages, *messages])
    trigger_tokens = min(provisioner.threshold_tokens, _effective_context_limit(provisioner, config))
    if tokens_before < trigger_tokens:
        return False
    provision = provisioner.compact(state.messages, force=True)
    if not provision.compacted:
        return False
    state.messages = provision.messages
    state.total_input_tokens = 0
    state.total_output_tokens = 0
    state.metadata["last_compaction"] = {
        "tokens_before": tokens_before,
        "tokens_after": overhead_tokens + provisioner.count_tokens([*provision.messages, *messages]),
        "request_overhead_tokens": overhead_tokens,
        "summarized_count": provision.summarized_count,
        "kept_count": provision.kept_count,
    }
    return True


def _provision_context_window(
    state: AgentState,
    *,
    config: dict[str, Any],
    model: str,
    force: bool = False,
) -> dict[str, Any]:
    """Compact current messages before the next provider request when needed.

    Unlike the cumulative spend budget, this checks the actual request window
    that will be sent to the provider. If compaction cannot reduce a context
    that is already over the model limit, callers must stop before making the
    next provider request.
    """
    provisioner = _compaction_provisioner(config=config, model=model)
    overhead_tokens = _request_overhead_tokens(config)
    effective_limit = _effective_context_limit(provisioner, config)
    message_tokens_before = provisioner.count_tokens(state.messages)
    tokens_before = overhead_tokens + message_tokens_before
    trigger_tokens = min(provisioner.threshold_tokens, effective_limit)
    if not force and tokens_before < trigger_tokens:
        return {
            "compacted": False,
            "blocked": False,
            "tokens_before": tokens_before,
            "tokens_after": tokens_before,
            "max_context_tokens": provisioner.max_context_tokens,
            "effective_context_tokens": effective_limit,
            "request_overhead_tokens": overhead_tokens,
            "reason": "below_threshold",
        }

    provision = provisioner.compact(state.messages, force=force or tokens_before >= effective_limit)
    if provision.compacted:
        state.messages = provision.messages
        state.total_input_tokens = 0
        state.total_output_tokens = 0
        tokens_after = overhead_tokens + provision.tokens_after
        state.metadata["last_compaction"] = {
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "request_overhead_tokens": overhead_tokens,
            "summarized_count": provision.summarized_count,
            "kept_count": provision.kept_count,
            "max_context_tokens": provisioner.max_context_tokens,
            "effective_context_tokens": effective_limit,
        }
        return {
            "compacted": True,
            "blocked": tokens_after >= effective_limit,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "max_context_tokens": provisioner.max_context_tokens,
            "effective_context_tokens": effective_limit,
            "request_overhead_tokens": overhead_tokens,
            "reason": provision.reason,
        }

    tokens_after = overhead_tokens + provision.tokens_after
    return {
        "compacted": False,
        "blocked": tokens_before >= effective_limit or tokens_after >= effective_limit,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "max_context_tokens": provisioner.max_context_tokens,
        "effective_context_tokens": effective_limit,
        "request_overhead_tokens": overhead_tokens,
        "reason": provision.reason,
        "error": provision.error,
    }


LLM_STREAM_RETRY_DELAYS = (5, 5, 5, 5, 5, 5)

_DEFAULT_TOOL_RESULT_SPILL_CHARS = 30_000
_DEFAULT_TOOL_RESULT_HEADROOM_CHARS = DEFAULT_HEADROOM_PREVIEW_CHARS
_CLAUDE_CODE_LEGACY_SHELL_TOOL_NAME = "".join(("Execute", "Shell"))
_CLAUDE_CODE_NATIVE_TOOL_NAMES = (
    "Bash",
    "BashOutput",
    "Edit",
    _CLAUDE_CODE_LEGACY_SHELL_TOOL_NAME,
    "ExitPlanMode",
    "Glob",
    "Grep",
    "KillBash",
    "LS",
    "MultiEdit",
    "NotebookEdit",
    "NotebookRead",
    "Read",
    "SlashCommand",
    "Task",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
    "Write",
)
_CLAUDE_CODE_CLI_DISALLOWED_TOOL_NAMES = (
    "Bash",
    "BashOutput",
    "Edit",
    _CLAUDE_CODE_LEGACY_SHELL_TOOL_NAME,
    "ExitPlanMode",
    "Glob",
    "Grep",
    "KillBash",
    "LS",
    "MultiEdit",
    "NotebookEdit",
    "NotebookRead",
    "Read",
    "Task",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
    "Write",
)
_CLAUDE_CODE_NATIVE_TOOL_KEYS = frozenset(
    re.sub(r"[^a-z0-9]+", "", name.lower()) for name in _CLAUDE_CODE_NATIVE_TOOL_NAMES
)
_CLAUDE_CODE_NATIVE_TOOL_DENY_ARG = ",".join(_CLAUDE_CODE_CLI_DISALLOWED_TOOL_NAMES)


def _session_token_budget(config: dict[str, Any]) -> int | None:
    """Return an explicit cumulative token budget, or ``None`` when uncapped."""
    value = config.get("max_budget_tokens")
    if value in (None, "", 0):
        return None
    if value is None:
        return None
    budget = int(value)
    return budget if budget > 0 else None


def _safe_tool_result_name(name: str) -> str:
    """Return a filesystem-safe tool-result path component."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip())[:48].strip("-")
    return safe or "tool"


def _project_root_for_runtime_memory(config: dict[str, Any]) -> Path:
    """Return the current project root used for runtime memory annotations."""
    raw = config.get("project_dir") or config.get("cwd") or config.get("workspace_root")
    if raw:
        return Path(str(raw)).expanduser()
    try:
        return Path.cwd()
    except OSError:
        return Path(".")


def _capture_runtime_workflow_memory(user_message: str, *, config: dict[str, Any], depth: int) -> None:
    """Persist explicit durable workflow notes before the provider sees the turn."""
    if depth != 0:
        return
    try:
        result = capture_user_workflow_memory(
            user_message,
            project_root=_project_root_for_runtime_memory(config),
        )
    except Exception as exc:
        logger.warning("Failed to capture runtime workflow memory: %s", exc)
        return
    if result.captured:
        logger.info("Captured runtime workflow memory in %s:%s", result.scope, result.path)


def _project_memory_for_tool_result(config: dict[str, Any]) -> tuple[Any | None, str]:
    """Return project-scoped agent memory, initializing it from config when needed."""
    from ..tools.agent_memory_tool import active_memory, set_active_memory

    memory = active_memory()
    if memory is not None and memory.has_project_scope():
        return memory, ""

    project_root = _project_root_for_runtime_memory(config)
    try:
        from ..runtime.agent_memory import AgentMemory

        global_dir = getattr(memory, "global_dir", None)
        if global_dir is None:
            initialized = AgentMemory(project_root=project_root)
        else:
            initialized = AgentMemory(project_root=project_root, global_dir=global_dir)
        initialized.ensure()
        set_active_memory(initialized)
        return initialized, ""
    except Exception as exc:
        return None, f"project agent memory could not initialize for {project_root}: {exc}"


def _spill_tool_result_to_project_memory(
    result: str,
    *,
    tool_name: str,
    tool_call_id: str,
    config: dict[str, Any],
    model: str,
) -> str:
    """Save an oversized tool result and return a compact pointer for context.

    The full result is preserved in project agent memory. The message returned
    to the provider contains a memory path plus a deterministic Headroom-style
    preview, and may also include an agent-written summary.
    """
    limit = int(config.get("tool_result_spill_chars") or _DEFAULT_TOOL_RESULT_SPILL_CHARS)
    if not isinstance(result, str) or len(result) <= limit:
        return result

    digest = hashlib.sha256(result.encode("utf-8", errors="replace")).hexdigest()[:12]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    safe_tool = _safe_tool_result_name(tool_name)
    safe_call = _safe_tool_result_name(tool_call_id)[:32]
    memory_path = f"tool-results/{stamp}-{safe_tool}-{safe_call}-{digest}.txt"
    saved = False
    save_error = ""
    try:
        memory, save_error = _project_memory_for_tool_result(config)
        if memory is not None:
            memory.write("project", memory_path, result)
            saved = True
    except Exception as exc:
        save_error = str(exc)

    summary = ""
    summary_error = ""
    summary_agent = compaction_summary_agent_from_config(model, config)
    if summary_agent is not None:
        try:
            summary = summary_agent(
                [
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": result,
                    }
                ],
                (
                    "Summarize this oversized tool result for the next model turn. "
                    "Preserve concrete findings, paths, errors, counts, and next actions. "
                    "Do not quote long raw output."
                ),
            ).strip()
        except Exception as exc:
            summary_error = str(exc)

    headroom_limit = int(config.get("tool_result_headroom_chars") or _DEFAULT_TOOL_RESULT_HEADROOM_CHARS)
    headroom = compress_tool_result(tool_name, result, max_chars=headroom_limit)

    lines = [
        "[Large tool result stored outside model context]",
        f"- Tool: `{tool_name}`",
        f"- Tool call id: `{tool_call_id}`",
        f"- Original size: {len(result):,} characters",
        f"- Headroom: {headroom.metadata_line()}",
    ]
    if saved:
        lines.append(f"- Full result: project agent memory `{memory_path}`")
        lines.append(f'- Read it with: `agent_memory_read("project", "{memory_path}")`')
    else:
        lines.append(
            f"- Full result was not inserted into context because {save_error or 'project memory unavailable'}."
        )
    if summary:
        lines.extend(["", "## Agent-written compact summary", summary])
    elif summary_error:
        lines.append(f"- Agent summary unavailable: {summary_error}")
    lines.extend(["", "## Built-in headroom preview", headroom.compressed])
    return "\n".join(lines)


def _append_model_visible_messages(
    state: AgentState,
    messages: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    model: str,
    compact_before_append: bool = True,
) -> bool:
    """Append provider-visible messages through one guarded runtime path.

    User-like messages can compact the existing prefix before they are appended.
    Tool-result messages deliberately skip pre-append compaction because their
    matching assistant ``tool_calls`` item must remain directly reachable until
    the result is appended; the post-tool context provisioner handles any
    follow-up compaction with tool-call repair.
    """
    if not messages:
        return False
    compacted = False
    if compact_before_append:
        compacted = _compact_before_append(state, messages, config=config, model=model)
    state.messages.extend(messages)
    return compacted


def _drain_steers_into_context(
    state: AgentState,
    *,
    steer_drain: Callable[[], list[str]] | None,
    config: dict[str, Any],
    model: str,
    label: str = "[mid-turn steer from user]",
) -> tuple[list[str], bool]:
    """Drain queued user steering into model-visible messages."""
    if steer_drain is None:
        return [], False
    pending = steer_drain()
    if not pending:
        return [], False
    joined = "\n\n".join(pending)
    compacted = _append_model_visible_messages(
        state,
        [{"role": "user", "content": f"{label}\n{joined}"}],
        config=config,
        model=model,
    )
    return pending, compacted


def _append_tool_result_message(
    state: AgentState,
    *,
    tool_name: str,
    tool_call_id: str,
    result: str,
    is_error: bool,
    config: dict[str, Any],
    model: str,
) -> str:
    """Spill and append a tool result while preserving tool-call pairing."""
    stored_result = _spill_tool_result_to_project_memory(
        result,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        config=config,
        model=model,
    )
    _append_model_visible_messages(
        state,
        [
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": stored_result,
                "is_error": is_error,
            }
        ],
        config=config,
        model=model,
        compact_before_append=False,
    )
    return stored_result


def _request_timeout(config: dict[str, Any]) -> float:
    """Return the provider request timeout in seconds."""
    raw = config.get("llm_timeout") or config.get("request_timeout") or os.environ.get("XERXES_LLM_TIMEOUT")
    try:
        return max(1.0, float(raw)) if raw is not None else 60.0
    except (TypeError, ValueError):
        return 60.0


def _request_connect_timeout(config: dict[str, Any]) -> float:
    """Return the provider connection timeout in seconds."""
    raw = config.get("connect_timeout") or os.environ.get("XERXES_LLM_CONNECT_TIMEOUT")
    try:
        return max(0.5, float(raw)) if raw is not None else 5.0
    except (TypeError, ValueError):
        return 5.0


def _request_max_retries(config: dict[str, Any], *, explicit_base_url: bool) -> int:
    """Return SDK retry count, defaulting custom endpoints to fail fast."""
    raw = config.get("max_retries") or os.environ.get("XERXES_LLM_MAX_RETRIES")
    if raw is None:
        return 0 if explicit_base_url else 2
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0 if explicit_base_url else 2


def run(
    user_message: str,
    state: AgentState,
    config: dict[str, Any],
    system_prompt: str,
    tool_executor: Callable[[str, dict[str, Any]], str] | None = None,
    tool_schemas: list[dict[str, Any]] | None = None,
    depth: int = 0,
    cancel_check: Callable[[], bool] | None = None,
    runtime_features_state: Any = None,
    steer_drain: Callable[[], list[str]] | None = None,
    agent_event_drain: Callable[[], list[str]] | None = None,
) -> Generator[StreamEvent, None, None]:
    """Drive a full turn: stream LLM output, run tools, and yield stream events.

    Appends the user message to ``state.messages``, then streams a response,
    parses thinking tags, records token usage and tool calls, gates each tool
    through the permission system, executes it, and feeds the result back.
    Exits on the first iteration that produces no tool calls. Set
    ``config["max_tool_turns"]`` or ``XERXES_MAX_TOOL_TURNS`` to a positive
    integer to add an explicit iteration ceiling.

    Args:
        user_message: Raw user text appended to the conversation.
        state: Session state. Mutated in place — messages, token totals,
            thinking content, and tool_executions are appended.
        config: Provider config dict (``model``, ``api_key``, ``permission_mode``,
            ``thinking``, sampling parameters, etc.).
        system_prompt: System prompt prepended to the request.
        tool_executor: Callable that runs a tool by name and returns its
            string result. When ``None``, the loop emits a stub message.
        tool_schemas: JSON-schema tool definitions exposed to the model.
        depth: Reserved subagent-depth marker (unused inside the loop).
        cancel_check: Optional poll invoked before each turn; returning
            ``True`` yields a ``[Cancelled]`` chunk and ends the loop.
        runtime_features_state: Optional runtime container holding an
            ``authoring_pipeline``; when present its ``on_turn_end`` is called
            and any authored skill is surfaced as :class:`SkillSuggestion`.
        steer_drain: Optional poll invoked between tool-loop iterations.
            Returns the list of pending steer strings to inject as a single
            user message before the next LLM request. Empty list ⇒ no-op.
        agent_event_drain: Optional poll invoked between tool-loop iterations.
            Returns rendered one-line summaries of sub-agent events
            (spawn / tool_start / done / …) accumulated since the last drain.
            Folded into a single ``user`` message so the main agent passively
            sees what its children are doing without having to poll
            ``CheckAgentMessages``. Empty list ⇒ no-op.

    Yields:
        Stream events in arrival order: :class:`TextChunk` /
        :class:`ThinkingChunk` while streaming, :class:`ToolStart`,
        optional :class:`PermissionRequest`, :class:`ToolEnd` per tool,
        :class:`TurnDone` at the end of every assistant turn, and
        :class:`SkillSuggestion` when authoring fires.
    """

    from xerxes.llms.registry import get_provider_config, resolve_provider

    config = dict(config)
    state.metadata["model"] = config.get("model", "")
    state.metadata.pop("last_connection_failure", None)
    _capture_runtime_workflow_memory(user_message, config=config, depth=depth)

    perm_mode = PermissionMode(config.get("permission_mode", "accept-all"))
    model = config.get("model", "")
    provider_name = resolve_provider(model, config)

    try:
        provider_cfg = get_provider_config(provider_name)
    except KeyError:
        provider_name = "openai"
        provider_cfg = get_provider_config("openai")

    config[_REQUEST_OVERHEAD_TOKENS_CONFIG_KEY] = estimate_request_overhead_tokens(
        model=model,
        system_prompt=system_prompt,
        tool_schemas=tool_schemas or [],
    )

    initial_user_message = {"role": "user", "content": user_message}
    precompacted_initial = _append_model_visible_messages(
        state,
        [initial_user_message],
        config=config,
        model=model,
    )
    if precompacted_initial:
        yield TextChunk("\n[Context compacted before adding the new turn.]\n")

    iteration_budget = iteration_budget_from_config(config)
    objective_guard_retries = 0
    objective_guard_limit = objective_guard_retry_limit(config)
    stopped_by_iteration_budget = False
    while True:
        if not iteration_budget.try_consume():
            stopped_by_iteration_budget = True
            break
        _turn = iteration_budget.used - 1
        if cancel_check and cancel_check():
            # Surface the cancellation so the caller can render a "stopped"
            # marker instead of treating the silent return as a clean finish.
            yield TextChunk("\n[Cancelled]")
            return

        pending_steers, steer_compacted = _drain_steers_into_context(
            state,
            steer_drain=steer_drain,
            config=config,
            model=model,
        )
        if pending_steers:
            if steer_compacted:
                yield TextChunk("\n[Context compacted before applying steer.]\n")
            yield TextChunk(f"\n[Steer applied: {pending_steers[0][:80]}{'…' if len(pending_steers[0]) > 80 else ''}]\n")

        context_provision = _provision_context_window(state, config=config, model=model)
        if context_provision["compacted"]:
            yield TextChunk(
                "\n[Context compacted before the next provider request "
                f"({context_provision['tokens_before']:,} → {context_provision['tokens_after']:,} tokens).]\n"
            )
        if context_provision["blocked"]:
            reason = str(context_provision.get("reason") or "compaction_unavailable")
            detail = str(context_provision.get("error") or "")
            suffix = f" ({detail})" if detail else ""
            yield TextChunk(
                "\n[Stopped: context window "
                f"({context_provision['tokens_after']:,}/{context_provision['max_context_tokens']:,} tokens) "
                f"exceeded and compaction could not reduce it: {reason}{suffix}.]"
            )
            yield TurnDone(
                input_tokens=0,
                output_tokens=0,
                tool_calls_count=0,
                model=model,
            )
            return

        budget_limit = _session_token_budget(config)
        cumulative = state.total_input_tokens + state.total_output_tokens
        if budget_limit is not None and cumulative >= budget_limit:
            yield TextChunk(
                f"\n[Stopped: session token budget ({budget_limit:,}) exhausted. "
                f"Used {cumulative:,} cumulative API tokens across {_turn} tool turns.]"
            )
            yield TurnDone(
                input_tokens=0,
                output_tokens=0,
                tool_calls_count=0,
                model=model,
            )
            return

        if agent_event_drain is not None:
            agent_lines = agent_event_drain()
            if agent_lines:
                agent_event_message = {
                    "role": "user",
                    "content": "[sub-agent events]\n" + "\n".join(agent_lines),
                }
                if _append_model_visible_messages(state, [agent_event_message], config=config, model=model):
                    yield TextChunk("\n[Context compacted before adding sub-agent events.]\n")

        state.turn_count += 1

        text = ""
        thinking_text = ""
        thinking_signature: str | None = None
        tool_calls: list[dict[str, Any]] = []
        in_tokens = 0
        out_tokens = 0
        cache_read_tokens = 0
        cache_creation_tokens = 0
        thinking_parser = _ThinkingParser()

        # Retry streaming failures with a fixed short delay.
        _MAX_RETRIES = len(LLM_STREAM_RETRY_DELAYS)
        _retry_attempt = 0
        _last_error: Exception | None = None
        _stream_succeeded = False
        _context_compaction_retry_attempted = False

        while _retry_attempt <= _MAX_RETRIES:
            try:
                _llm_gen = _stream_llm(
                    model=model,
                    provider_type=provider_cfg.type,
                    system=system_prompt,
                    messages=state.messages,
                    tool_schemas=tool_schemas or [],
                    config=config,
                )
                for chunk in _llm_gen:
                    if isinstance(chunk, TextChunk):
                        subs = thinking_parser.process(chunk.text)
                        for sub in subs:
                            if isinstance(sub, ThinkingChunk):
                                thinking_text += sub.text
                                yield sub
                            else:
                                text += sub.text
                                yield sub
                    elif isinstance(chunk, ThinkingChunk):
                        thinking_text += chunk.text
                        yield chunk
                    elif isinstance(chunk, dict):
                        tool_calls = chunk.get("tool_calls", [])
                        in_tokens = chunk.get("in_tokens", 0)
                        out_tokens = chunk.get("out_tokens", 0)
                        cache_read_tokens = chunk.get("cache_read_tokens", 0)
                        cache_creation_tokens = chunk.get("cache_creation_tokens", 0)
                        thinking_signature = chunk.get("thinking_signature")

                remaining = thinking_parser.process("")
                for sub in remaining:
                    if isinstance(sub, ThinkingChunk):
                        thinking_text += sub.text
                        yield sub
                _last_error = None
                _stream_succeeded = True
                break  # Success — exit retry loop
            except _RETRYABLE_STREAM_EXCEPTIONS as e:
                _last_error = e
                if _is_context_limit_error(e):
                    logger.error("LLM request exceeded the model context window: %s", e)
                    observed_context = _calibrate_request_overhead_from_context_error(
                        state,
                        config=config,
                        model=model,
                        error=e,
                    )
                    if text or thinking_text or tool_calls:
                        context_retry = {
                            "compacted": False,
                            "blocked": True,
                            "reason": "partial_response_started",
                            "error": "provider rejected the request after streaming partial output",
                        }
                    elif _context_compaction_retry_attempted:
                        context_retry = {
                            "compacted": False,
                            "blocked": True,
                            "reason": "retry_already_attempted",
                        }
                    else:
                        context_retry = _provision_context_window(state, config=config, model=model, force=True)
                        _context_compaction_retry_attempted = True
                        if context_retry["compacted"] and not context_retry["blocked"]:
                            yield TextChunk(
                                "\n[Context compacted after the provider rejected the request "
                                f"({context_retry['tokens_before']:,} → {context_retry['tokens_after']:,} tokens). "
                                "Retrying once.]\n"
                            )
                            continue
                    state.metadata["last_compaction_failure"] = {
                        **observed_context,
                        "reason": str(context_retry.get("reason") or "compaction_unavailable"),
                        "error": _short_context_error_detail(context_retry.get("error")),
                        "tokens_before": int(context_retry.get("tokens_before") or 0),
                        "tokens_after": int(context_retry.get("tokens_after") or 0),
                        "max_context_tokens": int(context_retry.get("max_context_tokens") or 0),
                    }
                    yield TextChunk(_format_context_limit_stop(provision=context_retry, observed=observed_context))
                    yield TurnDone(
                        input_tokens=in_tokens,
                        output_tokens=out_tokens,
                        tool_calls_count=0,
                        model=model,
                    )
                    return
                if _retry_attempt < _MAX_RETRIES:
                    delay = LLM_STREAM_RETRY_DELAYS[_retry_attempt]
                    logger.warning(
                        "LLM streaming error (attempt %d/%d): %s. Retrying in %ds...",
                        _retry_attempt + 1,
                        _MAX_RETRIES + 1,
                        e,
                        delay,
                    )
                    yield ProviderRetry(
                        error=str(e),
                        attempt=_retry_attempt + 1,
                        max_attempts=_MAX_RETRIES,
                        delay=delay,
                    )
                    time.sleep(delay)
                    _retry_attempt += 1
                else:
                    logger.error("LLM streaming failed after %d attempts: %s", _MAX_RETRIES + 1, e)
                    break  # All retries exhausted — fall through to error handling below

        # If all retries failed, handle the error gracefully
        if not _stream_succeeded and _last_error is not None:
            for sub in thinking_parser.process(""):
                if isinstance(sub, ThinkingChunk):
                    thinking_text += sub.text
                    yield sub
            if not _is_context_limit_error(_last_error):
                if not text and not thinking_text and state.messages and state.messages[-1].get("role") == "user":
                    content = state.messages[-1].get("content")
                    if content == user_message:
                        state.messages.pop()
                state.metadata["last_connection_failure"] = {
                    "user_message": user_message,
                    "error": str(_last_error),
                    "model": model,
                    "turn_count": state.turn_count,
                }
                yield ProviderRetry(
                    error=str(_last_error),
                    attempt=_MAX_RETRIES,
                    max_attempts=_MAX_RETRIES,
                    final=True,
                )
                if text or thinking_text:
                    msg_partial: dict[str, Any] = {"role": "assistant", "content": text, "tool_calls": []}
                    if thinking_text:
                        msg_partial["thinking"] = thinking_text
                        if thinking_signature:
                            msg_partial["thinking_signature"] = thinking_signature
                    state.messages.append(msg_partial)
                    if thinking_text:
                        state.thinking_content.append(thinking_text)
                yield TurnDone(
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    tool_calls_count=0,
                    model=model,
                )
                return
            observed_context = _calibrate_request_overhead_from_context_error(
                state,
                config=config,
                model=model,
                error=_last_error,
            )
            context_retry = _provision_context_window(state, config=config, model=model, force=True)
            state.metadata["last_compaction_failure"] = {
                **observed_context,
                "reason": str(context_retry.get("reason") or "compaction_unavailable"),
                "error": _short_context_error_detail(context_retry.get("error")),
                "tokens_before": int(context_retry.get("tokens_before") or 0),
                "tokens_after": int(context_retry.get("tokens_after") or 0),
                "max_context_tokens": int(context_retry.get("max_context_tokens") or 0),
            }
            yield TextChunk(_format_context_limit_stop(provision=context_retry, observed=observed_context))
            yield TurnDone(
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                tool_calls_count=0,
                model=model,
            )
            return

        # Skip appending an empty assistant turn (no visible text, no tool
        # calls, no thinking). messages_to_anthropic/messages_to_openai do not
        # serialise the 'thinking' key, so such a message would convert to an
        # empty content body and the provider would reject the next request
        # with HTTP 400.
        if text or tool_calls or thinking_text:
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": text,
                "tool_calls": tool_calls,
            }
            if thinking_text:
                msg["thinking"] = thinking_text
                if thinking_signature:
                    msg["thinking_signature"] = thinking_signature
            state.messages.append(msg)

        if thinking_text:
            state.thinking_content.append(thinking_text)
        elif state.thinking_content or text:
            state.thinking_content.append("")
        state.total_input_tokens += in_tokens
        state.total_output_tokens += out_tokens
        state.total_cache_read_tokens += cache_read_tokens
        state.total_cache_creation_tokens += cache_creation_tokens

        yield TurnDone(
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            tool_calls_count=len(tool_calls),
            model=model,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )

        if not tool_calls:
            objective_decision = inspect_objective_response(text, mode=config.get("mode", "code"))
            if objective_decision.should_continue:
                objective_guard_retries += 1
                if objective_guard_retries > objective_guard_limit:
                    yield TextChunk(
                        "\n[Stopped: objective guard could not get a verified completion or concrete blocker "
                        f"after {objective_guard_limit} retries. The last issue was: {objective_decision.reason}.]"
                    )
                    break
                if _append_model_visible_messages(
                    state,
                    [{"role": "user", "content": objective_decision.reminder}],
                    config=config,
                    model=model,
                ):
                    yield TextChunk("\n[Context compacted before adding objective guard reminder.]\n")
                yield TextChunk(f"\n[Objective gate: {objective_decision.reason}. Continuing.]\n")
                continue

        if runtime_features_state is not None and runtime_features_state.authoring_pipeline is not None:
            result = runtime_features_state.authoring_pipeline.on_turn_end(final_response=text)

            if result.authored:
                yield SkillSuggestion(
                    skill_name=result.skill_name,
                    version=result.version,
                    description="",
                    source_path=str(result.skill_path) if result.skill_path else "",
                    tool_count=len(result.candidate.events) if result.candidate else 0,
                    unique_tools=result.candidate.unique_tools if result.candidate else [],
                )

        if not tool_calls:
            late_steers, late_steer_compacted = _drain_steers_into_context(
                state,
                steer_drain=steer_drain,
                config=config,
                model=model,
                label="[steer from user saved for next turn]",
            )
            if late_steers:
                if late_steer_compacted:
                    yield TextChunk("\n[Context compacted before saving steer for next turn.]\n")
                yield TextChunk(
                    f"\n[Steer saved for next turn: {late_steers[0][:80]}{'…' if len(late_steers[0]) > 80 else ''}]\n"
                )
            break

        for tc_index, tc in enumerate(tool_calls):
            # Poll for cancellation BETWEEN tools so /cancel takes effect within the
            # turn instead of only at the next turn boundary. Every tool_use in the
            # assistant message needs a matching tool_result or the next request
            # 400s, so backfill synthetic cancelled-results for this and all
            # remaining (un-run) calls before returning.
            if cancel_check and cancel_check():
                for unrun_tc in tool_calls[tc_index:]:
                    pid = unrun_tc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
                    unrun_tc["id"] = pid
                    _append_model_visible_messages(
                        state,
                        [
                            {
                                "role": "tool",
                                "tool_call_id": pid,
                                "name": unrun_tc.get("name", ""),
                                "content": "[Cancelled by user before execution]",
                                "is_error": True,
                            }
                        ],
                        config=config,
                        model=model,
                        compact_before_append=False,
                    )
                yield TextChunk("\n[Cancelled]")
                return

            tc_id = tc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
            tc["id"] = tc_id
            tc_name = tc.get("name", "")
            tc_input = tc.get("input", {})

            # Validate tool arguments against schema before execution. This
            # catches common LLM mistakes (missing required params, wrong
            # types) early and gives the model actionable feedback to retry.
            _tool_schema = None
            for ts in tool_schemas or []:
                if ts.get("name") == tc_name:
                    _tool_schema = ts.get("input_schema")
                    break
            if _tool_schema:
                from xerxes.runtime.arg_validation import validate_and_format_error

                _arg_error = validate_and_format_error(tc_name, tc_input, _tool_schema)
                if _arg_error:
                    error_result = f"Error: {_arg_error}"
                    yield ToolStart(name=tc_name, inputs=tc_input, tool_call_id=tc_id)
                    yield ToolEnd(
                        name=tc_name,
                        result=error_result,
                        permitted=True,
                        tool_call_id=tc_id,
                        duration_ms=0.0,
                    )
                    _append_tool_result_message(
                        state,
                        tool_name=tc_name,
                        tool_call_id=tc_id,
                        result=error_result,
                        is_error=True,
                        config=config,
                        model=model,
                    )
                    continue

            yield ToolStart(name=tc_name, inputs=tc_input, tool_call_id=tc_id)

            permitted = check_permission(tc, perm_mode)
            if not permitted:
                req = PermissionRequest(
                    tool_name=tc_name,
                    description=format_permission_description(tc),
                    inputs=tc_input,
                )
                yield req
                permitted = req.granted

            duration_ms = 0.0
            is_error = not permitted
            if not permitted:
                result = "Denied: user rejected this operation."
            else:
                t0 = time.monotonic()
                if tool_executor:
                    try:
                        result = tool_executor(tc_name, tc_input)
                    except Exception as e:
                        result = f"Error executing {tc_name}: {e}"
                        is_error = True
                else:
                    result = f"Tool '{tc_name}' executed (no executor configured)."
                duration_ms = (time.monotonic() - t0) * 1000

            # A tool-level failure surfaced as an "Error:" string (FileEdit no-match,
            # non-zero shell, etc.) should also be flagged so the model recovers.
            if not is_error and isinstance(result, str) and result.lstrip()[:6].lower() == "error:":
                is_error = True

            result = _append_tool_result_message(
                state,
                tool_name=tc_name,
                tool_call_id=tc_id,
                result=result,
                is_error=is_error,
                config=config,
                model=model,
            )

            yield ToolEnd(
                name=tc_name,
                result=result,
                permitted=permitted,
                tool_call_id=tc_id,
                duration_ms=duration_ms,
            )
            state.tool_executions.append(
                {
                    "name": tc_name,
                    "inputs": tc_input,
                    "result": result,
                    "duration_ms": duration_ms,
                    "permitted": permitted,
                    "tool_call_id": tc_id,
                }
            )

        _inject_workspace_guard_message(state, config=config)
        post_tool_context = _provision_context_window(state, config=config, model=model)
        if post_tool_context["compacted"]:
            yield TextChunk(
                "\n[Context compacted after tool results "
                f"({post_tool_context['tokens_before']:,} → {post_tool_context['tokens_after']:,} tokens).]\n"
            )
        if post_tool_context["blocked"]:
            reason = str(post_tool_context.get("reason") or "compaction_unavailable")
            detail = str(post_tool_context.get("error") or "")
            suffix = f" ({detail})" if detail else ""
            yield TextChunk(
                "\n[Stopped: context window "
                f"({post_tool_context['tokens_after']:,}/{post_tool_context['max_context_tokens']:,} tokens) "
                f"exceeded after tool results and compaction could not reduce it: {reason}{suffix}.]"
            )
            yield TurnDone(
                input_tokens=0,
                output_tokens=0,
                tool_calls_count=0,
                model=model,
            )
            return
    if stopped_by_iteration_budget:
        # Without this branch the run silently stops mid-conversation and
        # looks identical to a normal completion.
        assert iteration_budget.max_iterations is not None
        yield TextChunk(
            "\n[Stopped: reached configured max tool turns "
            f"({iteration_budget.max_iterations}). Ask me to continue if there's more to do.]"
        )


def _inject_workspace_guard_message(state: AgentState, *, config: dict[str, Any]) -> None:
    """Feed risky working-tree state back to the model before its next step."""
    cwd = Path(str(config.get("project_dir") or os.getcwd())).expanduser()
    report = analyze_workspace_changes(cwd, state.tool_executions)
    if not report.should_notify:
        return
    metadata = dict(state.metadata or {})
    if metadata.get("last_change_guard_model_fingerprint") == report.fingerprint:
        return
    metadata["last_change_guard_model_fingerprint"] = report.fingerprint
    state.metadata = metadata
    _append_model_visible_messages(
        state,
        [
            {
                "role": "user",
                "content": (
                    "[Workspace guard]\n"
                    f"{format_change_guard_notification(report)}\n\n"
                    "Do not claim completion until the risky change is either fixed or explicitly justified, "
                    "and run focused verification for edited runtime/test/build surfaces."
                ),
            }
        ],
        config=config,
        model=str(config.get("model", "")),
    )


async def arun(
    user_message: str,
    state: AgentState,
    config: dict[str, Any],
    system_prompt: str,
    tool_executor: Callable[[str, dict[str, Any]], str] | None = None,
    tool_schemas: list[dict[str, Any]] | None = None,
    depth: int = 0,
    cancel_check: Callable[[], bool] | None = None,
    runtime_features_state: Any = None,
) -> AsyncGenerator[StreamEvent, None]:
    """Async adapter around :func:`run`.

    Runs ``next()`` on the sync generator inside the default thread executor
    so the event loop stays responsive while provider IO blocks. All arguments
    are forwarded unchanged to :func:`run`.

    Yields:
        The same stream events as :func:`run`, in order.
    """

    # arun is an async generator, so a running loop is guaranteed. Prefer
    # get_running_loop() — get_event_loop() is deprecated in 3.12+ and emits
    # a DeprecationWarning when no loop exists.
    loop = asyncio.get_running_loop()

    gen = run(
        user_message=user_message,
        state=state,
        config=config,
        system_prompt=system_prompt,
        tool_executor=tool_executor,
        tool_schemas=tool_schemas,
        depth=depth,
        cancel_check=cancel_check,
        runtime_features_state=runtime_features_state,
    )

    class _Sentinel:
        """Marker returned by ``next(..., _sentinel)`` when the generator is exhausted."""

        pass

    _sentinel = _Sentinel()
    while True:
        event = await loop.run_in_executor(None, lambda: next(gen, _sentinel))
        if isinstance(event, _Sentinel):
            break
        yield event


def _stream_llm(
    model: str,
    provider_type: str,
    system: str,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    config: dict[str, Any],
) -> Generator[TextChunk | ThinkingChunk | dict[str, Any], None, None]:
    """Dispatch streaming to the correct provider adapter.

    Yields ``TextChunk`` / ``ThinkingChunk`` while the stream is open and, at
    the end, a single dict with ``tool_calls`` and token counters.
    """

    from xerxes.llms.registry import PROVIDERS, provider_model, resolve_provider

    has_explicit_base = bool(config.get("base_url") or config.get("custom_base_url"))
    provider_name = resolve_provider(model, config)

    if has_explicit_base and provider_name not in PROVIDERS:
        model_name = model
        provider_name = "openai"
    else:
        model_name = provider_model(model, provider_name)

    if provider_type == "claude-code":
        yield from _stream_claude_code_cli(model_name, system, messages, tool_schemas, config)
    elif provider_type == "anthropic":
        yield from _stream_anthropic(model_name, system, messages, tool_schemas, config, provider_name)
    else:
        yield from _stream_openai_compat(model_name, system, messages, tool_schemas, config, provider_name)


def _claude_code_command(model: str, config: dict[str, Any]) -> list[str]:
    """Build the local Claude Code CLI command for provider-mode calls."""
    command = str(config.get("claude_code_command") or os.environ.get("CLAUDE_CODE_CLI") or "claude")
    model = model.strip()
    argv = [
        command,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
        "--disable-slash-commands",
        "--tools",
        "",
        "--disallowedTools",
        _CLAUDE_CODE_NATIVE_TOOL_DENY_ARG,
    ]
    if model and model not in {"default", "auto"}:
        argv.extend(["--model", model])
    effort = str(config.get("reasoning_effort") or "").strip().lower()
    if bool(config.get("thinking")) and effort and effort != "off":
        argv.extend(["--effort", effort])
    return argv


def _claude_code_env(config: dict[str, Any]) -> dict[str, str]:
    """Return environment for Claude Code subscription-mode subprocesses."""
    env = os.environ.copy()
    use_api_env = bool(config.get("claude_code_use_api_env")) or os.environ.get("XERXES_CLAUDE_CODE_USE_API_ENV") == "1"
    if not use_api_env:
        for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_TOKEN"):
            env.pop(key, None)
    return env


def _claude_code_auth_hint(text: str, model: str = "") -> str:
    """Append a focused hint for Claude Code auth failures."""
    lower = text.lower()
    if "invalid authentication credentials" not in lower and "failed to authenticate" not in lower:
        return text
    clean_model = model.strip()
    keys = [key for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "ANTHROPIC_AUTH_TOKEN") if os.environ.get(key)]
    if keys:
        joined = ", ".join(keys)
        return (
            f"{text}\n\n"
            "Claude Code auth hint: your shell has "
            f"{joined} set. Claude reports those as API-key auth instead of the claude.ai subscription. "
            "Unset them and restart the daemon with `uv run xerxes --refresh`, then launch Xerxes again. "
            "If the raw `claude -p` command still 401s after that, refresh Claude Code login with `claude auth login`."
        )
    if clean_model and clean_model not in {"default", "auto"}:
        return (
            f"{text}\n\n"
            f"Claude Code auth hint: raw Claude Code rejected the explicit model override `{clean_model}` with 401. "
            "This happened outside Xerxes too. Use `/model claude-code/default` to let Claude Code choose from "
            "its local config, or set a model id that your Claude Code login can run directly."
        )
    return (
        f"{text}\n\n"
        "Claude Code auth hint: raw Claude Code returned a 401 even without Anthropic API env vars. "
        "Refresh the local Claude Code login with `claude auth login`, then restart Xerxes with `uv run xerxes --refresh`."
    )


def _stringify_claude_code_content(content: Any) -> str:
    """Return a readable text form for a message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, default=str))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, default=str)


def _claude_code_prompt(
    system: str,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
) -> str:
    """Render Xerxes messages into the plain prompt Claude Code print mode accepts."""
    parts: list[str] = []
    if system:
        parts.append(f"# System\n{system.strip()}")
    if tool_schemas:
        tools = [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("input_schema", {}),
            }
            for tool in tool_schemas
        ]
        parts.append(
            "# Xerxes Tool Calling\n"
            "You are running as a Xerxes model provider, not as the Claude Code agent runtime. "
            "Native Claude Code tools are disabled and ignored. "
            f"Never call Bash, {_CLAUDE_CODE_LEGACY_SHELL_TOOL_NAME}, Read, Write, Edit, Grep, Glob, Task, "
            "TodoWrite, or other Claude Code-native tools. "
            "For shell work use Xerxes `exec_command`; poll, interrupt, or send input with `write_stdin`. "
            "When you need a Xerxes tool, output exactly one or more calls in this format and no markdown fence:\n"
            '<function=ToolName>{"arg":"value"}</function>\n'
            "After Xerxes executes the tool, the tool result will appear in the next message.\n"
            f"Available Xerxes tools:\n{json.dumps(tools, ensure_ascii=False, default=str)}"
        )

    rendered: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = _stringify_claude_code_content(message.get("content"))
        rendered.append(f"{role.upper()}:\n{content}".rstrip())
        if message.get("tool_calls"):
            rendered.append(
                "ASSISTANT_TOOL_CALLS:\n" + json.dumps(message.get("tool_calls"), ensure_ascii=False, default=str)
            )
        if message.get("tool_call_id"):
            rendered.append(f"TOOL_CALL_ID: {message.get('tool_call_id')}")
    if rendered:
        parts.append("# Conversation\n" + "\n\n".join(rendered))
    return "\n\n".join(parts).strip()


def _claude_code_text_from_event(event: dict[str, Any]) -> str:
    """Extract assistant text from a Claude Code stream-json event."""
    event_type = str(event.get("type", "")).lower()
    if "thinking" in event_type or "tool" in event_type:
        return ""

    if isinstance(event.get("delta"), dict):
        delta = event["delta"]
        text = delta.get("text") or delta.get("content")
        if isinstance(text, str):
            return text
    if isinstance(event.get("content"), str):
        return event["content"]

    message = event.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            return "".join(parts)

    result = event.get("result")
    if isinstance(result, str):
        return result
    return ""


def _strip_function_blocks(text: str) -> str:
    """Remove inline function-call blocks from assistant-visible text."""
    return re.sub(r"<function=[^>]+>.*?</function>", "", text, flags=re.S).strip()


def _parse_function_blocks(text: str) -> list[dict[str, Any]]:
    """Parse ``<function=name>{json}</function>`` blocks into Xerxes tool calls."""
    from xerxes.streaming.parsers.common import LlamaParser

    calls = []
    for index, call in enumerate(LlamaParser().parse(text)):
        calls.append(
            {
                "id": call.raw_id or f"call_cc_{index}",
                "name": call.name,
                "input": call.arguments,
            }
        )
    return calls


def _is_claude_code_native_tool_name(name: str) -> bool:
    """Return true for Claude Code-native tools that provider mode must ignore."""
    key = re.sub(r"[^a-z0-9]+", "", name.lower())
    return key in _CLAUDE_CODE_NATIVE_TOOL_KEYS


def _filter_claude_code_provider_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop Claude Code-native tool calls before they reach Xerxes execution."""
    filtered: list[dict[str, Any]] = []
    removed: list[str] = []
    for tool_call in tool_calls:
        name = str(tool_call.get("name") or "")
        if _is_claude_code_native_tool_name(name):
            removed.append(name)
            continue
        filtered.append(tool_call)
    if removed:
        logger.warning("Ignored Claude Code-native provider tool calls: %s", ", ".join(removed))
    return filtered


def _stream_claude_code_cli(
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    config: dict[str, Any],
) -> Generator[TextChunk | ThinkingChunk | dict[str, Any], None, None]:
    """Stream one provider-mode response through the local Claude Code CLI."""
    argv = _claude_code_command(model, config)
    executable = argv[0]
    resolved = executable if os.path.sep in executable else shutil.which(executable)
    if not resolved:
        yield TextChunk(
            "[Error: Claude Code CLI is not installed. Install it with `xerxes install --cloud-code`, "
            "then run `claude auth login`. If Claude Code is installed in a custom location, set "
            "CLAUDE_CODE_CLI to the executable path.]"
        )
        yield {"tool_calls": [], "in_tokens": 0, "out_tokens": 0}
        return
    argv[0] = resolved

    prompt = _claude_code_prompt(system, messages, tool_schemas)
    cwd = str(config.get("project_dir") or os.getcwd())
    text_parts: list[str] = []
    result_text = ""

    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_claude_code_env(config),
    )
    assert proc.stdin is not None
    proc.stdin.write(prompt)
    proc.stdin.close()
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            result_text += line + "\n"
            continue
        chunk_text = _claude_code_text_from_event(event)
        if not chunk_text:
            continue
        if event.get("type") == "result":
            result_text = chunk_text
            continue
        text_parts.append(chunk_text)

    stderr = proc.stderr.read() if proc.stderr is not None else ""
    return_code = proc.wait()
    raw_text = "".join(text_parts) or result_text
    if return_code != 0 and not raw_text.strip():
        error = _claude_code_auth_hint(stderr.strip() or f"Claude Code exited with status {return_code}", model)
        yield TextChunk(f"[Error: Claude Code provider failed: {error}]")
        yield {"tool_calls": [], "in_tokens": 0, "out_tokens": 0}
        return

    clean_text, marker_tool_calls = extract_assistant_tool_call_markers(raw_text, id_prefix="call_cc")
    tool_calls = _filter_claude_code_provider_tool_calls([*_parse_function_blocks(clean_text), *marker_tool_calls])
    visible_text = _claude_code_auth_hint(_strip_function_blocks(clean_text), model)
    if visible_text:
        yield TextChunk(visible_text)

    from xerxes.core.utils import estimate_messages_tokens, estimate_tokens

    yield {
        "tool_calls": tool_calls,
        "in_tokens": estimate_messages_tokens(messages) + estimate_tokens(system),
        "out_tokens": estimate_tokens(raw_text),
    }


def _stream_anthropic(
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    config: dict[str, Any],
    provider_name: str,
) -> Generator[TextChunk | ThinkingChunk | dict[str, Any], None, None]:
    """Stream from the Anthropic Messages API, with optional prompt caching.

    Wraps the system prompt and last tool schema with ``cache_control:
    ephemeral`` when caching is enabled, falls back to token-count estimation
    when the SDK reports zero usage, and dumps the request body to
    ``debug_request.json`` when ``config["debug"]`` is set.
    """

    try:
        import anthropic  # type: ignore
    except ImportError:
        yield TextChunk("[Error: anthropic package not installed]")
        return

    from xerxes.llms.registry import get_api_key

    api_key = get_api_key(provider_name, config)
    client = anthropic.Anthropic(api_key=api_key, timeout=_request_timeout(config))

    system_parts = [system] if system else []
    conversation_messages: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(m.get("content", ""))
        else:
            conversation_messages.append(m)
    combined_system = "\n\n".join(system_parts) if system_parts else ""

    api_messages = messages_to_anthropic(conversation_messages)

    # Prompt caching: when enabled (default on for Anthropic), wrap the system
    # prompt and the last tool schema with cache_control: ephemeral so the
    # request prefix is cacheable for ~5 minutes. /compact will change the
    # system prompt and invalidate the cache on the next turn — that's fine
    # and expected.
    use_caching = bool(config.get("prompt_caching", True))
    system_param: Any = combined_system
    tools_param = tool_schemas
    if use_caching:
        if combined_system:
            system_param = wrap_system_with_cache(combined_system)
        if tool_schemas:
            tools_param = wrap_tools_with_cache(tool_schemas)

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": config.get("max_tokens", 8192),
        "system": system_param,
        "messages": api_messages,
    }
    if tools_param:
        kwargs["tools"] = tools_param
    if config.get("thinking"):
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": config.get("thinking_budget", 10000),
        }

    if config.get("debug"):
        debug_payload = {
            "model": model,
            "messages": api_messages,
            "tools": tool_schemas,
            "max_tokens": kwargs.get("max_tokens"),
            "thinking": kwargs.get("thinking"),
        }
        import os as _os

        debug_path = _os.path.join(_os.getcwd(), "debug_request.json")
        try:
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(debug_payload, f, indent=2, ensure_ascii=False, default=str)
            logger.debug(
                "Anthropic request dumped to %s (%d messages, %d tools)",
                debug_path,
                len(api_messages),
                len(tool_schemas or []),
            )
        except Exception as e:
            logger.warning("Failed to dump debug request: %s", e)

    text = ""
    tool_calls: list[dict[str, Any]] = []

    with client.messages.stream(**kwargs) as stream:
        for event in stream:
            etype = getattr(event, "type", None)
            if etype == "content_block_delta":
                delta = event.delta
                dtype = getattr(delta, "type", None)
                if dtype == "text_delta":
                    text += delta.text
                    yield TextChunk(delta.text)
                elif dtype == "thinking_delta":
                    yield ThinkingChunk(delta.thinking)

        final = stream.get_final_message()
        thinking_signature: str | None = None
        for block in final.content:
            if block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
            elif block.type == "thinking":
                # Capture the signature so the thinking block can be replayed on a
                # later turn; Anthropic 400s on a text+tool_use turn that drops it.
                thinking_signature = getattr(block, "signature", None) or thinking_signature

        in_tok = getattr(final.usage, "input_tokens", 0) or 0
        out_tok = getattr(final.usage, "output_tokens", 0) or 0
        cache_read_tok, cache_creation_tok = extract_cache_tokens(final.usage)
        if in_tok == 0 or out_tok == 0:
            from xerxes.core.utils import estimate_messages_tokens

            if in_tok == 0:
                in_tok = estimate_messages_tokens(api_messages)
                if combined_system:
                    from xerxes.core.utils import estimate_tokens

                    in_tok += estimate_tokens(combined_system)
            if out_tok == 0:
                from xerxes.core.utils import estimate_tokens

                out_tok = estimate_tokens(text)
        yield {
            "tool_calls": tool_calls,
            "in_tokens": in_tok,
            "out_tokens": out_tok,
            "cache_read_tokens": cache_read_tok,
            "cache_creation_tokens": cache_creation_tok,
            "thinking_signature": thinking_signature,
        }


def _stream_openai_compat(
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    config: dict[str, Any],
    provider_name: str,
) -> Generator[TextChunk | ThinkingChunk | dict[str, Any], None, None]:
    """Stream from any OpenAI-compatible chat-completions endpoint.

    Applies per-provider quirks (MiniMax has to merge sequential user
    messages and rejects ``stream_options``), accumulates streamed
    ``tool_calls`` chunks by index, and emits ``reasoning_content`` deltas as
    :class:`ThinkingChunk` events.
    """

    try:
        from openai import BadRequestError, OpenAI
    except ImportError:
        yield TextChunk("[Error: openai package not installed]")
        return

    from xerxes.llms.registry import PROVIDERS, get_api_key, provider_default_headers

    api_key = config.get("api_key") or get_api_key(provider_name, config)
    prov = PROVIDERS.get(provider_name, PROVIDERS["openai"])

    explicit_base_url = bool(config.get("base_url") or config.get("custom_base_url"))
    base_url = config.get("base_url") or config.get("custom_base_url") or prov.base_url or "https://api.openai.com/v1"
    timeout = _request_timeout(config)
    client_kwargs: dict[str, Any] = {
        "api_key": api_key or "dummy",
        "base_url": base_url,
        "timeout": timeout,
        "max_retries": _request_max_retries(config, explicit_base_url=explicit_base_url),
    }
    # Some providers gate access by client identity (e.g. Kimi Code's
    # "Coding Agents only" allowlist returns 403 unless the request comes
    # from Kimi CLI / Claude Code / Roo Code / Kilo Code). Inject a matching
    # User-Agent so the request goes through.
    default_headers = provider_default_headers(provider_name)
    if default_headers:
        client_kwargs["default_headers"] = default_headers
    http_client = None
    if explicit_base_url:
        import httpx

        http_client = httpx.Client(
            timeout=httpx.Timeout(
                timeout,
                connect=min(timeout, _request_connect_timeout(config)),
            ),
            trust_env=False,
            headers=default_headers or None,
        )
        client_kwargs["http_client"] = http_client
    client = OpenAI(**client_kwargs)

    def _inner():
        oai_messages = messages_to_openai(messages, system=system)

        if provider_name == "minimax":
            normalized: list[dict[str, Any]] = []
            for msg in oai_messages:
                role = msg["role"]
                content = msg.get("content") or ""
                if role == "system":
                    role = "user"

                can_merge = (
                    normalized
                    and normalized[-1]["role"] == role
                    and role == "user"
                    and "tool_calls" not in msg
                    and "tool_call_id" not in msg
                    and "tool_calls" not in normalized[-1]
                    and "tool_call_id" not in normalized[-1]
                )
                if can_merge:
                    normalized[-1]["content"] += "\n\n" + content
                else:
                    normalized_msg: dict[str, Any] = {"role": role}

                    # Preserve content alongside tool_calls: an assistant turn may
                    # carry both narration text and tool_calls, and dropping the
                    # text corrupts the history MiniMax sees on later turns.
                    if "tool_calls" in msg:
                        normalized_msg["tool_calls"] = msg["tool_calls"]
                        if content:
                            normalized_msg["content"] = content
                    else:
                        normalized_msg["content"] = content
                    if "tool_call_id" in msg:
                        normalized_msg["tool_call_id"] = msg["tool_call_id"]
                        normalized_msg["content"] = content
                    normalized.append(normalized_msg)
            oai_messages = normalized

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
            "stream": True,
        }
        if tool_schemas:
            kwargs["tools"] = _tools_to_openai(tool_schemas)
            kwargs["tool_choice"] = "auto"
        if config.get("max_tokens"):
            kwargs["max_tokens"] = config["max_tokens"]
        for param in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
            if param in config:
                kwargs[param] = config[param]
        extra_body: dict[str, Any] = {}
        for param in ("top_k", "min_p", "repetition_penalty"):
            if param in config:
                extra_body[param] = config[param]
        if extra_body:
            kwargs["extra_body"] = extra_body

        # Reasoning effort (off/low/medium/high) for OpenAI-compatible providers that
        # honour it (OpenAI o-series, MiniMax, etc.). Set via the ``/thinking`` command.
        if config.get("thinking") and config.get("reasoning_effort"):
            kwargs["reasoning_effort"] = config["reasoning_effort"]

        if config.get("debug"):
            debug_payload = {
                "model": model,
                "base_url": base_url,
                "messages": oai_messages,
                "tools": kwargs.get("tools"),
                "sampling": {
                    k: kwargs[k]
                    for k in ("temperature", "top_p", "frequency_penalty", "presence_penalty", "max_tokens")
                    if k in kwargs
                },
            }
            import os as _os

            debug_path = _os.path.join(_os.getcwd(), "debug_request.json")
            try:
                with open(debug_path, "w", encoding="utf-8") as f:
                    json.dump(debug_payload, f, indent=2, ensure_ascii=False, default=str)
                logger.debug(
                    "OpenAI request dumped to %s (%d messages, %d tools)",
                    debug_path,
                    len(oai_messages),
                    len(kwargs.get("tools", [])),
                )
            except Exception as e:
                logger.warning("Failed to dump debug request: %s", e)

        text = ""
        tool_buf: dict[int, dict[str, Any]] = {}
        in_tok = out_tok = 0

        if provider_name not in {"minimax"}:
            kwargs["stream_options"] = {"include_usage": True}
        try:
            response_stream = client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            error_msg = str(e).lower()
            if "stream_options" in error_msg or "stream options" in error_msg:
                kwargs_without_stream = kwargs.copy()
                kwargs_without_stream.pop("stream_options", None)
                response_stream = client.chat.completions.create(**kwargs_without_stream)
            else:
                raise
        for chunk in response_stream:
            if not chunk.choices:
                if hasattr(chunk, "usage") and chunk.usage:
                    in_tok = chunk.usage.prompt_tokens or in_tok
                    out_tok = chunk.usage.completion_tokens or out_tok
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning is None:
                try:
                    raw = delta.model_extra or {}
                    reasoning = raw.get("reasoning_content")
                except (AttributeError, TypeError):
                    # Provider may omit model_extra; reasoning_content is non-standard/optional.
                    pass
            if reasoning is None:
                try:
                    raw_choice = chunk.model_extra or {}
                    reasoning = raw_choice.get("reasoning_content")
                except (AttributeError, TypeError):
                    # Provider may omit model_extra; reasoning_content is non-standard/optional.
                    pass
            if reasoning:
                yield ThinkingChunk(reasoning)

            if delta.content:
                text += delta.content
                yield TextChunk(delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_buf:
                        tool_buf[idx] = {"id": "", "name": "", "args": ""}
                    if tc.id:
                        tool_buf[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_buf[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_buf[idx]["args"] += tc.function.arguments

            if hasattr(chunk, "usage") and chunk.usage:
                in_tok = chunk.usage.prompt_tokens or in_tok
                out_tok = chunk.usage.completion_tokens or out_tok

        tool_calls: list[dict[str, Any]] = []
        for idx in sorted(tool_buf):
            v = tool_buf[idx]
            try:
                inp = json.loads(v["args"]) if v["args"] else {}
            except json.JSONDecodeError:
                inp = {"_raw": v["args"]}
            tool_calls.append(
                {
                    "id": v["id"] or f"call_{idx}",
                    "name": v["name"],
                    "input": inp,
                }
            )

        if in_tok == 0 or out_tok == 0:
            from xerxes.core.utils import estimate_messages_tokens

            if in_tok == 0:
                in_tok = estimate_messages_tokens(oai_messages)
            if out_tok == 0:
                from xerxes.core.utils import estimate_tokens

                out_tok = estimate_tokens(text)
        yield {
            "tool_calls": tool_calls,
            "in_tokens": in_tok,
            "out_tokens": out_tok,
        }

    try:
        yield from _inner()
    finally:
        if hasattr(client, "close"):
            client.close()
        if http_client is not None:
            http_client.close()


def _tools_to_openai(tool_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert neutral tool schemas to OpenAI ``{"type": "function", ...}`` form."""

    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        }
        for t in tool_schemas
    ]


__all__ = [
    "arun",
    "run",
]
