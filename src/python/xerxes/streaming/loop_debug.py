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
"""Debug-only sibling of :mod:`xerxes.streaming.loop`.

A stripped-down copy of the production loop kept for diagnostic comparisons:
no prompt caching, single ``<think>`` tag form, ``get_event_loop()`` instead
of ``get_running_loop()``, no cancellation surface, and no exhaustion warning
when ``MAX_TOOL_TURNS`` is reached. Useful for bisecting regressions caused by
the production loop's additional features.
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
from dataclasses import dataclass
from typing import Any

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
from .messages import messages_to_anthropic, messages_to_openai
from .permissions import (
    PermissionMode,
    check_permission,
    format_permission_description,
)


@dataclass
class _ThinkingParser:
    """Single-tag-form thinking splitter used by the debug loop.

    Only recognises ``<think>...</think>`` (no ``<thinking>`` variant).
    See :class:`xerxes.streaming.loop._ThinkingParser` for the production form.

    Attributes:
        _open_tag: Recognised opening tag.
        _close_tag: Recognised closing tag.
        _buffer: Unprocessed text carried across calls.
        _in_thinking: Whether we are currently inside a thinking block.
        _thinking_buf: Accumulated reasoning text awaiting emission.
    """

    _open_tag: str = "<think>"
    _close_tag: str = "</think>"
    _buffer: str = ""
    _in_thinking: bool = False
    _thinking_buf: str = ""

    def process(self, chunk_text: str) -> list[TextChunk | ThinkingChunk]:
        """Feed a streamed fragment and emit any completed chunks."""

        events: list[TextChunk | ThinkingChunk] = []
        self._buffer += chunk_text

        if not chunk_text and self._in_thinking:
            if self._thinking_buf:
                events.append(ThinkingChunk(self._thinking_buf))
            self._thinking_buf = ""
            self._buffer = ""
            self._in_thinking = False
            return events

        while self._buffer:
            if not self._in_thinking:
                idx = self._buffer.find(self._open_tag)
                if idx == -1:
                    if self._buffer:
                        events.append(TextChunk(self._buffer))
                        self._buffer = ""
                    break
                if idx > 0:
                    events.append(TextChunk(self._buffer[:idx]))
                self._buffer = self._buffer[idx + len(self._open_tag) :]
                self._in_thinking = True
                self._thinking_buf = ""

            else:
                idx = self._buffer.find(self._close_tag)
                if idx == -1:
                    self._thinking_buf += self._buffer
                    self._buffer = ""
                    break
                if idx > 0:
                    self._thinking_buf += self._buffer[:idx]
                self._buffer = self._buffer[idx + len(self._close_tag) :]
                self._in_thinking = False
                if self._thinking_buf:
                    events.append(ThinkingChunk(self._thinking_buf))
                    self._thinking_buf = ""

        return events


def _parse_thinking_tags(
    text: str,
) -> tuple[str, str]:
    """Split a complete blob into ``(visible, thinking)``; debug-loop variant."""

    thinking_pattern = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL)
    thinking_parts = thinking_pattern.findall(text)
    thinking_text = "".join(thinking_parts).strip()
    visible_text = thinking_pattern.sub("", text).strip()
    return visible_text, thinking_text


logger = logging.getLogger(__name__)

MAX_TOOL_TURNS = 50


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
) -> Generator[StreamEvent, None, None]:
    """Diagnostic copy of :func:`xerxes.streaming.loop.run` without caching.

    Same semantics as the production loop but without prompt caching, the
    ``<thinking>`` tag variant, max-turns warning, or cache-token accounting.
    Useful for isolating issues introduced by those features.
    """

    from xerxes.llms.registry import detect_provider, get_provider_config

    state.messages.append({"role": "user", "content": user_message})
    state.metadata["model"] = config.get("model", "")

    perm_mode = PermissionMode(config.get("permission_mode", "auto"))
    model = config.get("model", "")
    provider_name = detect_provider(model)

    try:
        provider_cfg = get_provider_config(provider_name)
    except KeyError:
        provider_name = "openai"
        provider_cfg = get_provider_config("openai")

    for _turn in range(MAX_TOOL_TURNS):
        if cancel_check and cancel_check():
            return

        state.turn_count += 1

        text = ""
        thinking_text = ""
        tool_calls: list[dict[str, Any]] = []
        in_tokens = 0
        out_tokens = 0
        thinking_parser = _ThinkingParser()

        try:
            for chunk in _stream_llm(
                model=model,
                provider_type=provider_cfg.type,
                system=system_prompt,
                messages=state.messages,
                tool_schemas=tool_schemas or [],
                config=config,
            ):
                if isinstance(chunk, TextChunk):
                    for sub in thinking_parser.process(chunk.text):
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
        except Exception as e:
            logger.error("LLM streaming error: %s", e)
            yield TextChunk(f"\n[Error: {e}]")
            return

        remaining = thinking_parser.process("")
        for sub in remaining:
            if isinstance(sub, ThinkingChunk):
                thinking_text += sub.text
                yield sub

        msg: dict[str, Any] = {
            "role": "assistant",
            "content": text,
            "tool_calls": tool_calls,
        }
        if thinking_text:
            msg["thinking"] = thinking_text
        state.messages.append(msg)

        if thinking_text:
            state.thinking_content.append(thinking_text)
        elif state.thinking_content or text:
            state.thinking_content.append("")
        state.total_input_tokens += in_tokens
        state.total_output_tokens += out_tokens

        yield TurnDone(
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            tool_calls_count=len(tool_calls),
            model=model,
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

        for tc in tool_calls:
            tc_id = tc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
            tc["id"] = tc_id
            tc_name = tc.get("name", "")
            tc_input = tc.get("input", {})

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
            if not permitted:
                result = "Denied: user rejected this operation."
                yield ToolEnd(
                    name=tc_name,
                    result=result,
                    permitted=False,
                    tool_call_id=tc_id,
                )
            else:
                t0 = time.monotonic()
                if tool_executor:
                    try:
                        result = tool_executor(tc_name, tc_input)
                    except Exception as e:
                        result = f"Error executing {tc_name}: {e}"
                else:
                    result = f"Tool '{tc_name}' executed (no executor configured)."
                duration_ms = (time.monotonic() - t0) * 1000

                yield ToolEnd(
                    name=tc_name,
                    result=result,
                    permitted=True,
                    tool_call_id=tc_id,
                    duration_ms=duration_ms,
                )
            state.tool_executions.append(
                {
                    "name": tc_name,
                    "inputs": tc_input,
                    "result": result,
                    "duration_ms": duration_ms if permitted else 0.0,
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
                }
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
    """Async wrapper around the debug :func:`run`; uses the legacy ``get_event_loop``."""

    loop = asyncio.get_event_loop()

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
        """End-of-iterator marker returned by ``next(..., _sentinel)``."""

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
    """Dispatch to the correct debug-loop provider adapter."""

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
    """Stream from Anthropic without prompt caching (debug-loop variant)."""

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

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": config.get("max_tokens", 8192),
        "system": combined_system,
        "messages": api_messages,
    }
    if tool_schemas:
        kwargs["tools"] = tool_schemas
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
            print(
                f"\n  [DEBUG] Request dumped to {debug_path} ({len(api_messages)} messages, {len(tool_schemas or [])} tools)"
            )
        except Exception as e:
            print(f"\n  [DEBUG] Failed to dump request: {e}")

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
        for block in final.content:
            if block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        yield {
            "tool_calls": tool_calls,
            "in_tokens": final.usage.input_tokens,
            "out_tokens": final.usage.output_tokens,
        }


def _stream_openai_compat(
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    config: dict[str, Any],
    provider_name: str,
) -> Generator[TextChunk | ThinkingChunk | dict[str, Any], None, None]:
    """Stream from an OpenAI-compatible endpoint (debug-loop variant)."""

    try:
        from openai import BadRequestError, OpenAI
    except ImportError:
        yield TextChunk("[Error: openai package not installed]")
        return

    from xerxes.llms.registry import PROVIDERS, get_api_key

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
    if explicit_base_url:
        import httpx

        client_kwargs["http_client"] = httpx.Client(
            timeout=httpx.Timeout(
                timeout,
                connect=min(timeout, _request_connect_timeout(config)),
            ),
            trust_env=False,
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

                if "tool_calls" in msg:
                    normalized_msg["tool_calls"] = msg["tool_calls"]
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
            logger.info("DEBUG: request dumped to %s", debug_path)
            print(
                f"\n  [DEBUG] Request dumped to {debug_path} ({len(oai_messages)} messages, {len(kwargs.get('tools', []))} tools)"
            )
        except Exception as e:
            print(f"\n  [DEBUG] Failed to dump request: {e}")

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
            except Exception:
                pass
        if reasoning is None:
            try:
                raw_choice = chunk.model_extra or {}
                reasoning = raw_choice.get("reasoning_content")
            except Exception:
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

    yield {
        "tool_calls": tool_calls,
        "in_tokens": in_tok,
        "out_tokens": out_tok,
    }


def _tools_to_openai(tool_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert neutral tool schemas to OpenAI ``{"type": "function"}`` form."""

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
