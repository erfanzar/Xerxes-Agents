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
"""Messages module for Xerxes.

Exports:
    - messages_to_anthropic
    - messages_to_openai
    - messages_from_anthropic
    - messages_from_openai"""

from __future__ import annotations

import json
from typing import Any, TypeAlias

NeutralMessage: TypeAlias = dict[str, Any]


def messages_to_anthropic(messages: list[NeutralMessage]) -> list[dict[str, Any]]:
    """Messages to anthropic.

    Args:
        messages (list[NeutralMessage]): IN: messages. OUT: Consumed during execution.
    Returns:
        list[dict[str, Any]]: OUT: Result of the operation."""

    result: list[dict[str, Any]] = []
    """Messages to anthropic.

    Args:
        messages (list[NeutralMessage]): IN: messages. OUT: Consumed during execution.
    Returns:
        list[dict[str, Any]]: OUT: Result of the operation."""
    """Messages to anthropic.

    Args:
        messages (list[NeutralMessage]): IN: messages. OUT: Consumed during execution.
    Returns:
        list[dict[str, Any]]: OUT: Result of the operation."""
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
    """Messages to openai.

    Args:
        messages (list[NeutralMessage]): IN: messages. OUT: Consumed during execution.
        system (str | None, optional): IN: system. Defaults to None. OUT: Consumed during execution.
    Returns:
        list[dict[str, Any]]: OUT: Result of the operation."""

    result: list[dict[str, Any]] = []
    """Messages to openai.

    Args:
        messages (list[NeutralMessage]): IN: messages. OUT: Consumed during execution.
        system (str | None, optional): IN: system. Defaults to None. OUT: Consumed during execution.
    Returns:
        list[dict[str, Any]]: OUT: Result of the operation."""
    """Messages to openai.

    Args:
        messages (list[NeutralMessage]): IN: messages. OUT: Consumed during execution.
        system (str | None, optional): IN: system. Defaults to None. OUT: Consumed during execution.
    Returns:
        list[dict[str, Any]]: OUT: Result of the operation."""

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
    """Messages from anthropic.

    Args:
        messages (list[dict[str, Any]]): IN: messages. OUT: Consumed during execution.
    Returns:
        list[NeutralMessage]: OUT: Result of the operation."""

    result: list[NeutralMessage] = []
    """Messages from anthropic.

    Args:
        messages (list[dict[str, Any]]): IN: messages. OUT: Consumed during execution.
    Returns:
        list[NeutralMessage]: OUT: Result of the operation."""
    """Messages from anthropic.

    Args:
        messages (list[dict[str, Any]]): IN: messages. OUT: Consumed during execution.
    Returns:
        list[NeutralMessage]: OUT: Result of the operation."""

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
    """Messages from openai.

    Args:
        messages (list[dict[str, Any]]): IN: messages. OUT: Consumed during execution.
    Returns:
        list[NeutralMessage]: OUT: Result of the operation."""

    result: list[NeutralMessage] = []
    """Messages from openai.

    Args:
        messages (list[dict[str, Any]]): IN: messages. OUT: Consumed during execution.
    Returns:
        list[NeutralMessage]: OUT: Result of the operation."""
    """Messages from openai.

    Args:
        messages (list[dict[str, Any]]): IN: messages. OUT: Consumed during execution.
    Returns:
        list[NeutralMessage]: OUT: Result of the operation."""

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
