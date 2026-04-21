# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""Generator-based streaming agent loop.

Inspired by the nano-claude-code agent loop, this module implements a
multi-turn agent loop as a Python generator. The loop:

"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncGenerator, Callable, Generator
from typing import Any


class _ThinkingParser:
    """Stateful parser that splits text chunks on <think> / <thinking> XML tags.

    Models that emit <think> ... </think> or <thinking> ... </thinking>
    as plain text (e.g. DeepSeek) rather than as a dedicated
    reasoning_content field need this to separate thinking from the
    visible response.
    """

    __slots__ = ("_buffer", "_in_thinking", "_thinking_buf")

    _OPEN_TAGS: tuple[str, ...] = ("<think>", "<thinking>")
    _CLOSE_TAGS: tuple[str, ...] = ("</think>", "</thinking>")

    def __init__(self) -> None:
        self._buffer = ""
        self._in_thinking = False
        self._thinking_buf = ""

    @staticmethod
    def _find_any(text: str, tags: tuple[str, ...]) -> tuple[int, str]:
        """Return (index, matched_tag) for the earliest tag found, or (-1, '')."""
        earliest_idx = -1
        earliest_tag = ""
        for tag in tags:
            idx = text.find(tag)
            if idx != -1 and (earliest_idx == -1 or idx < earliest_idx):
                earliest_idx = idx
                earliest_tag = tag
        return earliest_idx, earliest_tag

    def process(self, chunk_text: str) -> list[TextChunk | ThinkingChunk]:
        """Parse a text fragment and return a list of TextChunk or ThinkingChunk events."""
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
                idx, tag = self._find_any(self._buffer, self._OPEN_TAGS)
                if idx == -1:
                    if self._buffer:
                        events.append(TextChunk(self._buffer))
                        self._buffer = ""
                    break
                if idx > 0:
                    events.append(TextChunk(self._buffer[:idx]))
                self._buffer = self._buffer[idx + len(tag) :]
                self._in_thinking = True
                self._thinking_buf = ""

            else:
                idx, tag = self._find_any(self._buffer, self._CLOSE_TAGS)
                if idx == -1:
                    self._thinking_buf += self._buffer
                    self._buffer = ""
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
    """Split a complete text block into (visible_text, thinking_text).

    Both segments are stripped of the XML tags. If no thinking tags are
    found, thinking_text is empty.
    """
    thinking_pattern = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL)
    thinking_parts = thinking_pattern.findall(text)
    thinking_text = "".join(thinking_parts).strip()
    visible_text = thinking_pattern.sub("", text).strip()
    return visible_text, thinking_text


from .events import (  # noqa: E402
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
from .messages import messages_to_anthropic, messages_to_openai  # noqa: E402
from .permissions import (  # noqa: E402
    PermissionMode,
    check_permission,
    format_permission_description,
)

logger = logging.getLogger(__name__)

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
    runtime_features_state: Any = None,
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

            remaining = thinking_parser.process("")
            for sub in remaining:
                if isinstance(sub, ThinkingChunk):
                    thinking_text += sub.text
                    yield sub
        except Exception as e:
            logger.error("LLM streaming error: %s", e)
            yield TextChunk(f"\n[Error: {e}]")
            return

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
            tc_id = tc.get("id", "")
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
                truncated_result = result
                yield ToolEnd(
                    name=tc_name,
                    result=truncated_result,
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
                truncated_result = result[:2000] if len(result) > 2000 else result

                yield ToolEnd(
                    name=tc_name,
                    result=truncated_result,
                    permitted=True,
                    tool_call_id=tc_id,
                    duration_ms=duration_ms,
                )
            state.tool_executions.append(
                {
                    "name": tc_name,
                    "inputs": tc_input,
                    "result": truncated_result,
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
                    "content": truncated_result,
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
    """Async variant of :func:`run`.

    Same interface as :func:`run` but yields events asynchronously.
    The tool executor can be sync or async.
    """
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
    """Stream from the appropriate LLM provider.

    Yields :class:`TextChunk` and :class:`ThinkingChunk` events during
    streaming, and a final dict with ``tool_calls``, ``in_tokens``,
    ``out_tokens`` at the end.
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
    """Stream from the Anthropic API."""
    try:
        import anthropic
    except ImportError:
        yield TextChunk("[Error: anthropic package not installed]")
        return

    from xerxes.llms.registry import get_api_key

    api_key = get_api_key(provider_name, config)
    client = anthropic.Anthropic(api_key=api_key)

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

        in_tok = getattr(final.usage, "input_tokens", 0) or 0
        out_tok = getattr(final.usage, "output_tokens", 0) or 0
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
        from openai import BadRequestError, OpenAI
    except ImportError:
        yield TextChunk("[Error: openai package not installed]")
        return

    from xerxes.llms.registry import PROVIDERS, get_api_key

    api_key = config.get("api_key") or get_api_key(provider_name, config)
    prov = PROVIDERS.get(provider_name, PROVIDERS["openai"])

    base_url = config.get("base_url") or config.get("custom_base_url") or prov.base_url or "https://api.openai.com/v1"
    client = OpenAI(api_key=api_key or "dummy", base_url=base_url)

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
