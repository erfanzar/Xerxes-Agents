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
"""Mistral tool-call parser."""

from __future__ import annotations

import json
import re

from . import ParsedToolCall, ToolCallParser


class MistralParser(ToolCallParser):
    """Parser for the Mistral ``[TOOL_CALLS][...]`` JSON-array format.

    Unlike the tag-based parsers, Mistral emits the bare token
    ``[TOOL_CALLS]`` followed by a JSON array of objects with ``name`` and
    ``arguments`` (or ``parameters``) fields. Each entry may also carry an
    ``id`` that is preserved on :attr:`ParsedToolCall.raw_id`.
    """

    name = "mistral"

    def parse(self, text: str) -> list[ParsedToolCall]:
        """Extract Mistral tool calls from a ``[TOOL_CALLS][...]`` JSON array."""
        match = re.search(r"\[TOOL_CALLS\]\s*(\[.*?\])", text, re.S)
        if not match:
            return []
        try:
            arr = json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            return []
        out: list[ParsedToolCall] = []
        if not isinstance(arr, list):
            return out
        for entry in arr:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            args = entry.get("arguments") or entry.get("parameters") or {}
            if not isinstance(args, dict):
                args = {}
            if name:
                out.append(ParsedToolCall(name=str(name), arguments=args, raw_id=str(entry.get("id", ""))))
        return out
