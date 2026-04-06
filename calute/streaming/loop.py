# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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


"""Generator-based streaming agent loop.

Inspired by the nano-claude-code agent loop, this module implements a
multi-turn agent loop as a Python generator. The loop:

1. Sends the conversation to the LLM provider (streaming).
2. Yields :class:`TextChunk` / :class:`ThinkingChunk` events as tokens arrive.
3. When the LLM returns tool calls, yields :class:`ToolStart` events.
4. Checks permissions via the permission system.
5. Executes approved tools and yields :class:`ToolEnd` events.
6. Appends tool results to the conversation and loops back to step 1.
7. When no more tool calls are returned, the loop ends.

The loop is designed to be consumed by any frontend (TUI, API server, web UI)
through a simple ``for event in run(...):`` pattern.

Usage::

    from calute.streaming.loop import run
    from calute.streaming.events import AgentState, TextChunk, ToolStart

    state = AgentState()
    config = {"model": "gpt-4o", "permission_mode": "auto"}

    for event in run("List files in /tmp", state, config, system_prompt="..."):
        match event:
            case TextChunk(text=t):
                print(t, end="", flush=True)
            case ToolStart(name=n):
                print(f"\\n→ {n}")

Async variant::

    async for event in arun("List files", state, config, system_prompt="..."):
        ...
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator, Callable, Generator
from typing import Any

from .events import (
    AgentState,
    PermissionRequest,
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

logger = logging.getLogger(__name__)

# Maximum number of consecutive tool-use turns before the loop breaks
# (prevents infinite loops when the model keeps calling tools)
MAX_TOOL_TURNS = 50


def run(
    user_message: str,
    state: AgentState,
    config: dict[str, Any],
    system_prompt: str,
    tool_executor: Callable[[str, dict[str, Any]], str] | None = None,
    tool_schemas: list[dict[str, Any]] | None = None,
    depth: int = 0,
    cancel_check: Callable[[], bool] | None = None,
) -> Generator[StreamEvent, None, None]:
    """Multi-turn streaming agent loop (synchronous generator).

    Yields :data:`StreamEvent` instances as the agent processes the
    user message, calls tools, and produces responses.

    Args:
        user_message: The user's input message.
        state: Mutable :class:`AgentState` tracking conversation history
            and token counts.
        config: Configuration dict. Expected keys:

            - ``model`` (str): Model name for the LLM.
            - ``permission_mode`` (str): One of ``"auto"``, ``"accept-all"``,
              ``"manual"``. Defaults to ``"auto"``.
            - ``max_tokens`` (int): Max tokens per LLM turn.
            - ``thinking`` (bool): Enable extended thinking.
            - ``thinking_budget`` (int): Token budget for thinking.

        system_prompt: System prompt for the LLM.
        tool_executor: Callable ``(tool_name, tool_input) -> result_string``.
            If None, tool calls are acknowledged but not executed.
        tool_schemas: List of tool schemas (Anthropic format) to send to
            the LLM. If None, no tools are available.
        depth: Sub-agent nesting depth (0 for top-level).
        cancel_check: Optional callable returning ``True`` to abort early.

    Yields:
        :data:`StreamEvent` instances.
    """
    from calute.llms.registry import detect_provider, get_provider_config

    # Append user turn
    state.messages.append({"role": "user", "content": user_message})
    state.metadata["model"] = config.get("model", "")

    perm_mode = PermissionMode(config.get("permission_mode", "auto"))
    model = config.get("model", "")
    provider_name = detect_provider(model)

    # If explicit base_url and unknown provider, fall back to openai-compat
    try:
        provider_cfg = get_provider_config(provider_name)
    except KeyError:
        provider_name = "openai"
        provider_cfg = get_provider_config("openai")

    for _turn in range(MAX_TOOL_TURNS):
        if cancel_check and cancel_check():
            return

        state.turn_count += 1

        # ── Stream from LLM ────────────────────────────────────────
        text = ""
        thinking_text = ""
        tool_calls: list[dict[str, Any]] = []
        in_tokens = 0
        out_tokens = 0

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
                    text += chunk.text
                    yield chunk
                elif isinstance(chunk, ThinkingChunk):
                    thinking_text += chunk.text
                    yield chunk
                elif isinstance(chunk, dict):
                    # Completed turn info
                    tool_calls = chunk.get("tool_calls", [])
                    in_tokens = chunk.get("in_tokens", 0)
                    out_tokens = chunk.get("out_tokens", 0)
        except Exception as e:
            logger.error("LLM streaming error: %s", e)
            yield TextChunk(f"\n[Error: {e}]")
            return

        # Record assistant turn (including thinking content)
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": text,
            "tool_calls": tool_calls,
        }
        if thinking_text:
            msg["thinking"] = thinking_text
        state.messages.append(msg)

        # Store thinking content in state for session persistence
        if thinking_text:
            state.thinking_content.append(thinking_text)
        elif state.thinking_content or text:
            # Pad with empty string to keep indexing aligned with turns
            state.thinking_content.append("")
        state.total_input_tokens += in_tokens
        state.total_output_tokens += out_tokens

        yield TurnDone(
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            tool_calls_count=len(tool_calls),
            model=model,
        )

        if not tool_calls:
            break  # No tools → done

        # ── Execute tools ────────────────────────────────────────────
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            tc_name = tc.get("name", "")
            tc_input = tc.get("input", {})

            yield ToolStart(name=tc_name, inputs=tc_input, tool_call_id=tc_id)

            # Permission gate
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

            # Record full tool execution metadata in state
            state.tool_executions.append(
                {
                    "name": tc_name,
                    "inputs": tc_input,
                    "result": result[:2000] if len(result) > 2000 else result,
                    "duration_ms": duration_ms if permitted else 0.0,
                    "permitted": permitted,
                    "tool_call_id": tc_id,
                }
            )

            # Append tool result to messages
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
) -> AsyncGenerator[StreamEvent, None]:
    """Async variant of :func:`run`.

    Same interface as :func:`run` but yields events asynchronously.
    The tool executor can be sync or async.
    """
    # Delegate to sync generator in a thread for now.
    # A fully native async implementation can replace this later.
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
    )

    sentinel = object()
    while True:
        event = await loop.run_in_executor(None, next, gen, sentinel)
        if event is sentinel:
            break
        yield event  # type: ignore[misc]


# ── Internal LLM streaming adapter ────────────────────────────────────────


def _stream_llm(
    model: str,
    provider_type: str,
    system: str,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    config: dict[str, Any],
) -> Generator[TextChunk | ThinkingChunk | dict[str, Any], None, None]:
    """Stream from the appropriate LLM provider.

    Yields :class:`TextChunk` and :class:`ThinkingChunk` events during
    streaming, and a final dict with ``tool_calls``, ``in_tokens``,
    ``out_tokens`` at the end.
    """
    from calute.llms.registry import PROVIDERS, bare_model, detect_provider

    # When a base_url is explicitly set, treat the model name as-is
    # (don't strip provider prefix — it may be a HuggingFace-style model ID)
    has_explicit_base = bool(config.get("base_url") or config.get("custom_base_url"))
    provider_name = detect_provider(model)

    if has_explicit_base and provider_name not in PROVIDERS:
        # Unknown provider prefix (e.g. "erfanzar/jeffery-27b") — use full model name
        model_name = model
        provider_name = "openai"  # treat as generic OpenAI-compatible
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
    """Stream from the Anthropic API."""
    try:
        import anthropic
    except ImportError:
        yield TextChunk("[Error: anthropic package not installed]")
        return

    from calute.llms.registry import get_api_key

    api_key = get_api_key(provider_name, config)
    client = anthropic.Anthropic(api_key=api_key)

    api_messages = messages_to_anthropic(messages)

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": config.get("max_tokens", 8192),
        "system": system,
        "messages": api_messages,
    }
    if tool_schemas:
        kwargs["tools"] = tool_schemas
    if config.get("thinking"):
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": config.get("thinking_budget", 10000),
        }

    # Debug: dump the full request payload
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
    """Stream from any OpenAI-compatible API."""
    try:
        from openai import OpenAI
    except ImportError:
        yield TextChunk("[Error: openai package not installed]")
        return

    from calute.llms.registry import PROVIDERS, get_api_key

    api_key = config.get("api_key") or get_api_key(provider_name, config)
    prov = PROVIDERS.get(provider_name, PROVIDERS["openai"])

    base_url = config.get("base_url") or config.get("custom_base_url") or prov.base_url or "https://api.openai.com/v1"
    client = OpenAI(api_key=api_key or "dummy", base_url=base_url)

    oai_messages = messages_to_openai(messages, system=system)

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
    # Pass through sampling parameters
    for param in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
        if param in config:
            kwargs[param] = config[param]
    # top_k, min_p, repetition_penalty are not standard OpenAI params
    # but some providers (vLLM, etc.) accept them via extra_body
    extra_body: dict[str, Any] = {}
    for param in ("top_k", "min_p", "repetition_penalty"):
        if param in config:
            extra_body[param] = config[param]
    if extra_body:
        kwargs["extra_body"] = extra_body

    # Debug: dump the full request payload
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

    response_stream = client.chat.completions.create(**kwargs)
    for chunk in response_stream:
        if not chunk.choices:
            if hasattr(chunk, "usage") and chunk.usage:
                in_tok = chunk.usage.prompt_tokens or in_tok
                out_tok = chunk.usage.completion_tokens or out_tok
            continue

        choice = chunk.choices[0]
        delta = choice.delta

        # Handle thinking/reasoning content (reasoning_content field)
        # Try attribute first, then fall back to dict-style access for
        # vLLM/custom servers that add extra fields the SDK doesn't model
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
    """Convert Anthropic-style tool schemas to OpenAI function-calling format."""
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
