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
"""Live context-window accounting for UI and daemon status."""

from __future__ import annotations

import json
from typing import Any

from .token_counter import SmartTokenCounter


def request_scaffolding_messages(
    *,
    system_prompt: str = "",
    tool_schemas: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return pseudo-messages for provider request overhead outside chat history."""
    scaffolding: list[dict[str, Any]] = []
    if system_prompt:
        scaffolding.append({"role": "system", "content": system_prompt})
    if tool_schemas:
        scaffolding.append(
            {
                "role": "system",
                "content": "[available tool schemas]\n"
                + json.dumps(tool_schemas, ensure_ascii=False, default=str, sort_keys=True),
            }
        )
    return scaffolding


def estimate_request_overhead_tokens(
    *,
    model: str,
    system_prompt: str = "",
    tool_schemas: list[dict[str, Any]] | None = None,
) -> int:
    """Return estimated request tokens that are not stored in ``state.messages``."""
    scaffolding = request_scaffolding_messages(system_prompt=system_prompt, tool_schemas=tool_schemas)
    if not scaffolding:
        return 0
    return _count_messages(scaffolding, model=model)


def estimate_context_tokens(
    messages: list[dict[str, Any]],
    *,
    model: str,
    system_prompt: str = "",
    tool_schemas: list[dict[str, Any]] | None = None,
) -> int:
    """Return the estimated token count for messages currently in the prompt window."""
    request_messages = [
        *request_scaffolding_messages(system_prompt=system_prompt, tool_schemas=tool_schemas),
        *messages,
    ]
    if not request_messages:
        return 0
    return _count_messages(request_messages, model=model)


def _count_messages(messages: list[dict[str, Any]], *, model: str) -> int:
    """Count messages, falling back to character-based estimation."""
    if not messages:
        return 0
    try:
        return max(0, SmartTokenCounter(model=model).count_tokens(messages))
    except Exception:
        text = "\n".join(f"{message.get('role', '')}: {message.get('content', '')}" for message in messages)
        return max(0, len(text) // 4)
