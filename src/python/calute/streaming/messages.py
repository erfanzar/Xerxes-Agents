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


"""Neutral message format with bidirectional provider converters.

Inspired by the nano-claude-code pattern, this module defines a
provider-agnostic message representation and conversion functions for
the Anthropic and OpenAI API formats.

Neutral format
--------------

All messages are plain dicts with a ``role`` key:

- **User**::

      {"role": "user", "content": "Hello"}

- **Assistant**::

      {"role": "assistant", "content": "Hi!", "tool_calls": [
          {"id": "tc_1", "name": "Read", "input": {"file_path": "/foo"}}
      ]}

- **Tool result**::

      {"role": "tool", "tool_call_id": "tc_1", "name": "Read",
       "content": "file contents..."}

The :class:`NeutralMessage` type alias documents the expected dict shape.
The converter functions handle all the quirks of each API's message format.

Usage::

    from calute.streaming.messages import messages_to_anthropic, messages_to_openai

    neutral = [
        {"role": "user", "content": "Read /etc/hosts"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "tc_1", "name": "Read", "input": {"file_path": "/etc/hosts"}}
        ]},
        {"role": "tool", "tool_call_id": "tc_1", "name": "Read",
         "content": "127.0.0.1 localhost"},
    ]

    anthropic_msgs = messages_to_anthropic(neutral)
    openai_msgs    = messages_to_openai(neutral)
"""

from __future__ import annotations

import json
from typing import Any, TypeAlias

NeutralMessage: TypeAlias = dict[str, Any]


def messages_to_anthropic(messages: list[NeutralMessage]) -> list[dict[str, Any]]:
    """Convert neutral messages to the Anthropic API format.

    Anthropic's API requires:
    - Assistant messages with tool calls to use ``tool_use`` content blocks.
    - Tool results to be wrapped in ``tool_result`` content blocks inside
      a ``user`` message (consecutive tool results are merged).

    Args:
        messages: List of neutral-format messages.

    Returns:
        List of messages formatted for the Anthropic Messages API.
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
    """Convert neutral messages to the OpenAI Chat Completions API format.

    OpenAI's API requires:
    - Tool calls to be encoded as ``function`` objects with JSON-serialized
      ``arguments``.
    - Tool results to use ``role: "tool"`` with ``tool_call_id``.
    - System prompt as a separate ``system`` role message.

    Args:
        messages: List of neutral-format messages.
        system: Optional system prompt to prepend.

    Returns:
        List of messages formatted for the OpenAI Chat Completions API.
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

    return result


def messages_from_anthropic(messages: list[dict[str, Any]]) -> list[NeutralMessage]:
    """Convert Anthropic API format messages back to neutral format.

    Handles both string content and content-block arrays.

    Args:
        messages: List of Anthropic-format messages.

    Returns:
        List of neutral-format messages.
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
    """Convert OpenAI API format messages back to neutral format.

    Args:
        messages: List of OpenAI-format messages.

    Returns:
        List of neutral-format messages.
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
