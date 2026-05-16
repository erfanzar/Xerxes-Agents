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
"""Anthropic prompt-caching helpers.

Anthropic caches a request prefix when the request contains explicit
``cache_control`` markers. We mark the system prompt and the last tool
definition; together they constitute the stable prefix that survives across
turns until compaction changes the system prompt.

Anthropic allows up to 4 cache breakpoints per request. We use 2: one on the
system prompt, one on the tools block. That leaves room for a future
"stable conversation prefix" marker (Plan 03 compression).
"""

from __future__ import annotations

from typing import Any

# Models whose providers expose Anthropic-style ephemeral cache_control.
# Currently only Anthropic itself; OpenAI/Gemini have their own caching
# mechanisms (Predicted Outputs / context caching) handled separately.
SUPPORTS_CACHING: tuple[str, ...] = ("anthropic",)


def wrap_system_with_cache(system_text: str) -> str | list[dict[str, Any]]:
    """Return a system value that Anthropic will cache.

    Anthropic accepts either a string or a list of content blocks for
    the ``system`` parameter. When given blocks, the final block can
    carry ``cache_control: {"type": "ephemeral"}`` to make the entire
    system prompt cacheable. If the text is empty, we return the empty
    string so the request body matches the default uncached shape."""

    if not system_text:
        return ""
    return [
        {
            "type": "text",
            "text": system_text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def wrap_tools_with_cache(tool_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach ``cache_control`` to the *last* tool schema.

    Anthropic caches the entire tools block when the final tool entry
    bears a cache_control marker. Modifies a copy of the list so the
    caller's schemas are untouched."""

    if not tool_schemas:
        return tool_schemas
    out: list[dict[str, Any]] = [dict(t) for t in tool_schemas]
    # Drop cache_control from earlier entries (defensive — we only want
    # one breakpoint at the tail).
    for t in out[:-1]:
        t.pop("cache_control", None)
    out[-1]["cache_control"] = {"type": "ephemeral"}
    return out


def extract_cache_tokens(usage: Any) -> tuple[int, int]:
    """Return ``(cache_read_tokens, cache_creation_tokens)`` from a usage object.

    Accepts the Anthropic SDK's ``Usage`` model or any duck-typed object
    with attributes/keys named ``cache_read_input_tokens`` and
    ``cache_creation_input_tokens``. Missing fields return 0."""

    if usage is None:
        return 0, 0

    def _get(name: str) -> int:
        v = getattr(usage, name, None)
        if v is None and isinstance(usage, dict):
            v = usage.get(name)
        return int(v or 0)

    return _get("cache_read_input_tokens"), _get("cache_creation_input_tokens")


__all__ = [
    "SUPPORTS_CACHING",
    "extract_cache_tokens",
    "wrap_system_with_cache",
    "wrap_tools_with_cache",
]
