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
until the model stops requesting tools or :data:`MAX_TOOL_TURNS` is exhausted.

:func:`arun` adapts the sync generator to an async generator by running
``next()`` in the default executor. ``_stream_anthropic`` and
``_stream_openai_compat`` are the provider adapters; the rest are private
helpers for thinking-tag splitting, prompt caching, and request-config
parsing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Generator
from typing import Any

from ..context.compaction_provisioner import (
    CompactionProvisioner,
    compaction_summary_agent_from_config,
)
from ..llms.registry import get_context_limit
from .events import (
    AgentState,
    PermissionRequest,
    SkillSuggestion,
    StreamEvent,
    TextChunk,
    ThinkingChunk,
    ToolEnd,
    ToolStart,
    TurnDone,
)
from .messages import bound_tool_result, messages_to_anthropic, messages_to_openai
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


def _compaction_provisioner(
    *,
    config: dict[str, Any],
    model: str,
    budget_limit: int | None = None,
) -> CompactionProvisioner:
    """Build the shared compaction provisioner for the streaming loop."""
    context_limit = int(
        config.get("max_context_tokens") or get_context_limit(model) or budget_limit or _DEFAULT_TOKEN_BUDGET
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
    provision = provisioner.compact_before_append(state.messages, messages)
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


MAX_TOOL_TURNS = 50

# Default token budget when none is specified in config. Prevents runaway
# agents from burning unlimited tokens on infinite tool-call loops.
_DEFAULT_TOKEN_BUDGET = 500_000


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

    Appends the user message to ``state.messages``, then loops up to
    :data:`MAX_TOOL_TURNS` times: stream a response, parse thinking tags,
    record token usage and tool calls, gate each tool through the permission
    system, execute it, and feed the result back. Exits on the first
    iteration that produces no tool calls.

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

    from xerxes.llms.registry import detect_provider, get_provider_config

    state.metadata["model"] = config.get("model", "")

    perm_mode = PermissionMode(config.get("permission_mode", "accept-all"))
    model = config.get("model", "")
    provider_name = detect_provider(model)

    try:
        provider_cfg = get_provider_config(provider_name)
    except KeyError:
        provider_name = "openai"
        provider_cfg = get_provider_config("openai")

    initial_user_message = {"role": "user", "content": user_message}
    precompacted_initial = _compact_before_append(
        state,
        [initial_user_message],
        config=config,
        model=model,
    )
    if precompacted_initial:
        yield TextChunk("\n[Context compacted before adding the new turn.]\n")
    state.messages.append(initial_user_message)

    for _turn in range(MAX_TOOL_TURNS):
        if cancel_check and cancel_check():
            # Surface the cancellation so the caller can render a "stopped"
            # marker instead of treating the silent return as a clean finish.
            yield TextChunk("\n[Cancelled]")
            return

        budget_limit = config.get("max_budget_tokens") or _DEFAULT_TOKEN_BUDGET
        cumulative = state.total_input_tokens + state.total_output_tokens
        if cumulative >= budget_limit:
            # Try to compact context before giving up
            compacted = _try_compact_messages(state, budget_limit, config=config, model=model)
            if compacted:
                yield TextChunk("\n[Context compacted to free up tokens. Continuing...]")
                # Recalculate after compaction
                cumulative = state.total_input_tokens + state.total_output_tokens
                if cumulative < budget_limit:
                    continue  # Retry the turn with compacted context

            yield TextChunk(
                f"\n[Stopped: token budget ({budget_limit:,}) exhausted. "
                f"Used {cumulative:,} tokens across {_turn} tool turns.]"
            )
            yield TurnDone(
                input_tokens=0,
                output_tokens=0,
                tool_calls_count=0,
                model=model,
            )
            return

        if steer_drain is not None:
            pending = steer_drain()
            if pending:
                joined = "\n\n".join(pending)
                steer_message = {
                    "role": "user",
                    "content": f"[mid-turn steer from user]\n{joined}",
                }
                if _compact_before_append(state, [steer_message], config=config, model=model):
                    yield TextChunk("\n[Context compacted before applying steer.]\n")
                state.messages.append(steer_message)
                yield TextChunk(f"\n[Steer applied: {pending[0][:80]}{'…' if len(pending[0]) > 80 else ''}]\n")

        if agent_event_drain is not None:
            agent_lines = agent_event_drain()
            if agent_lines:
                agent_event_message = {
                    "role": "user",
                    "content": "[sub-agent events]\n" + "\n".join(agent_lines),
                }
                if _compact_before_append(state, [agent_event_message], config=config, model=model):
                    yield TextChunk("\n[Context compacted before adding sub-agent events.]\n")
                state.messages.append(agent_event_message)

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

        # Retry configuration: 3 attempts with exponential backoff (5s, 10s, 20s)
        _MAX_RETRIES = 3
        _RETRY_DELAYS = [5, 10, 20]
        _retry_attempt = 0
        _last_error: Exception | None = None

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
                break  # Success — exit retry loop
            except Exception as e:
                _last_error = e
                if _retry_attempt < _MAX_RETRIES:
                    delay = _RETRY_DELAYS[_retry_attempt]
                    logger.warning(
                        "LLM streaming error (attempt %d/%d): %s. Retrying in %ds...",
                        _retry_attempt + 1,
                        _MAX_RETRIES + 1,
                        e,
                        delay,
                    )
                    yield TextChunk(
                        f"\n[Error: the response was interrupted: {e}. Retrying in {delay}s... ({_retry_attempt + 1}/{_MAX_RETRIES})]"
                    )
                    time.sleep(delay)
                    _retry_attempt += 1
                else:
                    logger.error("LLM streaming failed after %d attempts: %s", _MAX_RETRIES + 1, e)
                    break  # All retries exhausted — fall through to error handling below

        # If all retries failed, handle the error gracefully
        if _last_error is not None and _retry_attempt >= _MAX_RETRIES:
            for sub in thinking_parser.process(""):
                if isinstance(sub, ThinkingChunk):
                    thinking_text += sub.text
                    yield sub
            err_note = f"[Error: the response was interrupted: {_last_error}]"
            combined = (f"{text}\n\n{err_note}" if text else err_note).strip()
            msg_err: dict[str, Any] = {"role": "assistant", "content": combined, "tool_calls": []}
            if thinking_text:
                msg_err["thinking"] = thinking_text
                if thinking_signature:
                    msg_err["thinking_signature"] = thinking_signature
            state.messages.append(msg_err)
            if thinking_text:
                state.thinking_content.append(thinking_text)
            yield TextChunk(f"\n{err_note}")
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
                    state.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": pid,
                            "name": unrun_tc.get("name", ""),
                            "content": "[Cancelled by user before execution]",
                            "is_error": True,
                        }
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
                    yield ToolStart(name=tc_name, inputs=tc_input, tool_call_id=tc_id)
                    yield ToolEnd(
                        name=tc_name,
                        result=f"Error: {_arg_error}",
                        permitted=True,
                        tool_call_id=tc_id,
                        duration_ms=0.0,
                    )
                    state.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": tc_name,
                            "content": f"Error: {_arg_error}",
                            "is_error": True,
                        }
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

            # Clamp oversized output so one tool can't poison the whole window.
            result = bound_tool_result(result)
            # A tool-level failure surfaced as an "Error:" string (FileEdit no-match,
            # non-zero shell, etc.) should also be flagged so the model recovers.
            if not is_error and isinstance(result, str) and result.lstrip()[:6].lower() == "error:":
                is_error = True

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

            state.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": tc_name,
                    "content": result,
                    "is_error": is_error,
                }
            )
    else:
        # The for-else fires when we exhausted MAX_TOOL_TURNS without ever
        # hitting the ``break`` that fires on "no more tool calls". Without
        # this branch the run silently stops mid-conversation and looks
        # identical to a normal completion.
        yield TextChunk(
            f"\n[Stopped: reached max tool turns ({MAX_TOOL_TURNS}). Ask me to continue if there's more to do.]"
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

    from xerxes.llms.registry import PROVIDERS, bare_model, detect_provider

    has_explicit_base = bool(config.get("base_url") or config.get("custom_base_url"))
    provider_name = detect_provider(model)

    if has_explicit_base and provider_name not in PROVIDERS:
        model_name = model
        provider_name = "openai"
    else:
        model_name = bare_model(model)

    if provider_type == "anthropic":
        yield from _stream_anthropic(model_name, system, messages, tool_schemas, config, provider_name)
    else:
        yield from _stream_openai_compat(model_name, system, messages, tool_schemas, config, provider_name)


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
        import anthropic
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
    if explicit_base_url:
        import httpx

        client_kwargs["http_client"] = httpx.Client(
            timeout=httpx.Timeout(
                timeout,
                connect=min(timeout, _request_connect_timeout(config)),
            ),
            trust_env=False,
            headers=default_headers or None,
        )
    client = OpenAI(**client_kwargs)

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
    except BadRequestError:
        if kwargs.pop("stream_options", None) is not None:
            response_stream = client.chat.completions.create(**kwargs)
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
