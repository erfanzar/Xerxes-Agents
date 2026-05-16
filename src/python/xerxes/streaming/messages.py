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
"""Neutral <-> provider message conversion for the streaming loop.

The agent loop stores conversation state in a neutral list of dicts shaped
as ``{"role": str, "content": str, "tool_calls": [...], "tool_call_id": str}``.
These helpers translate to and from the Anthropic Messages and OpenAI
Chat-Completions schemas so the loop body remains provider-agnostic.
"""

from __future__ import annotations

import json
from typing import Any, TypeAlias

NeutralMessage: TypeAlias = dict[str, Any]


def messages_to_anthropic(messages: list[NeutralMessage]) -> list[dict[str, Any]]:
    """Convert neutral messages to Anthropic Messages-API content blocks.

    Assistant messages become a block list combining a ``text`` block (if any
    content) with one ``tool_use`` block per call. Consecutive ``tool`` role
    messages collapse into a single ``user`` message whose content is a list
    of ``tool_result`` blocks, matching Anthropic's expected interleaving.

    Args:
        messages: Neutral conversation history.

    Returns:
        Anthropic-format message list ready for the ``messages`` API field.
    """

    result: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        role = m["role"]

        if role == "user":
            result.append({"role": "user", "content": m["content"]})
            i += 1

        elif role == "assistant":
            blocks: list[dict[str, Any]] = []
            text = m.get("content", "")
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in m.get("tool_calls", []):
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["input"],
                    }
                )
            result.append({"role": "assistant", "content": blocks})
            i += 1

        elif role == "tool":
            tool_blocks: list[dict[str, Any]] = []
            while i < len(messages) and messages[i]["role"] == "tool":
                t = messages[i]
                tool_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": t["tool_call_id"],
                        "content": t["content"],
                    }
                )
                i += 1
            result.append({"role": "user", "content": tool_blocks})

        else:
            i += 1

    return result


def messages_to_openai(
    messages: list[NeutralMessage],
    system: str | None = None,
) -> list[dict[str, Any]]:
    """Convert neutral messages to OpenAI chat-completions form.

    Tool calls are serialised as ``{"id", "type": "function", "function":
    {"name", "arguments": json}}``. ``tool`` role messages carry the
    ``tool_call_id`` linking them back to the originating call.

    Args:
        messages: Neutral conversation history.
        system: Optional system prompt prepended as the first message.

    Returns:
        OpenAI-format message list ready for ``client.chat.completions.create``.
    """

    result: list[dict[str, Any]] = []

    if system:
        result.append({"role": "system", "content": system})

    for m in messages:
        role = m["role"]

        if role == "user":
            result.append({"role": "user", "content": m["content"]})

        elif role == "assistant":
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": m.get("content") or None,
            }
            tcs = m.get("tool_calls", [])
            if tcs:
                msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["input"], ensure_ascii=False),
                        },
                    }
                    for tc in tcs
                ]
            result.append(msg)

        elif role == "tool":
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": m["tool_call_id"],
                    "content": m["content"],
                }
            )

        elif role == "system":
            result.append({"role": "system", "content": m["content"]})

    return result


def messages_from_anthropic(messages: list[dict[str, Any]]) -> list[NeutralMessage]:
    """Convert Anthropic content-block messages back to the neutral format.

    Text blocks join into a single content string; ``tool_use`` blocks become
    ``tool_calls`` entries on the assistant message; ``tool_result`` blocks
    become separate ``tool`` messages with the original ``tool_use_id``.

    Args:
        messages: Anthropic-format message list.

    Returns:
        Neutral conversation history.
    """

    result: list[NeutralMessage] = []

    for m in messages:
        role = m["role"]
        content = m.get("content", "")

        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []

            for block in content:
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block["text"])
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "id": block["id"],
                            "name": block["name"],
                            "input": block["input"],
                        }
                    )
                elif btype == "tool_result":
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "name": block.get("name", ""),
                            "content": block.get("content", ""),
                        }
                    )

            if tool_calls:
                result.append(
                    {
                        "role": "assistant",
                        "content": "\n".join(text_parts),
                        "tool_calls": tool_calls,
                    }
                )
            elif tool_results:
                result.extend(tool_results)
            else:
                result.append({"role": role, "content": "\n".join(text_parts)})

    return result


def messages_from_openai(messages: list[dict[str, Any]]) -> list[NeutralMessage]:
    """Convert OpenAI chat-completions messages back to the neutral format.

    ``system`` role is mapped to ``user`` because the neutral loop carries the
    system prompt separately. Assistant ``tool_calls`` are JSON-decoded into
    parsed argument dicts to match the neutral schema.

    Args:
        messages: OpenAI-format message list.

    Returns:
        Neutral conversation history.
    """

    result: list[NeutralMessage] = []

    for m in messages:
        role = m.get("role", "")

        if role == "system":
            result.append({"role": "user", "content": m["content"]})

        elif role == "user":
            result.append({"role": "user", "content": m["content"]})

        elif role == "assistant":
            msg: NeutralMessage = {
                "role": "assistant",
                "content": m.get("content") or "",
            }
            tcs = m.get("tool_calls", [])
            if tcs:
                msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"]["arguments"]),
                    }
                    for tc in tcs
                ]
            result.append(msg)

        elif role == "tool":
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": m["tool_call_id"],
                    "name": m.get("name", ""),
                    "content": m.get("content", ""),
                }
            )

    return result


__all__ = [
    "NeutralMessage",
    "messages_from_anthropic",
    "messages_from_openai",
    "messages_to_anthropic",
    "messages_to_openai",
]
