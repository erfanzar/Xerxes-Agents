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
"""Shared raw-text parsing helpers plus the XML-tag and Llama parsers."""

from __future__ import annotations

import json
import re
from typing import Any

from . import ParsedToolCall, ToolCallParser


def _safe_json(text: str) -> dict[str, Any] | None:
    """Decode JSON, returning ``None`` for syntax errors or non-dict roots."""
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def parse_tool_call_blocks(text: str, *, open_tag: str, close_tag: str) -> list[ParsedToolCall]:
    """Extract every JSON tool-call block delimited by the given tag pair.

    Accepts the common shape ``<open>{"name", "arguments"}</close>`` plus
    its aliases (``function``/``tool`` for the name, ``input``/``parameters``
    for the arguments). Blocks whose JSON fails to decode are skipped.

    Locating payloads is done by index-based scanning (``str.find`` for the
    open tag, then :meth:`json.JSONDecoder.raw_decode` to consume exactly one
    JSON value). This is linear time and immune to the close tag appearing
    inside a JSON string value, unlike a non-greedy regex which truncates the
    payload at the first literal close tag and degrades quadratically on
    repeated open tags.

    Args:
        text: Raw model output.
        open_tag: Literal opening tag (e.g. ``"<tool_call>"``).
        close_tag: Literal closing tag (e.g. ``"</tool_call>"``).

    Returns:
        Parsed tool calls in source order.
    """
    decoder = json.JSONDecoder()
    out: list[ParsedToolCall] = []
    pos = 0
    n = len(text)
    while True:
        start = text.find(open_tag, pos)
        if start < 0:
            break
        idx = start + len(open_tag)
        while idx < n and text[idx].isspace():
            idx += 1
        # Only an object payload can carry a tool call, so skip openers not
        # followed by '{'. This also keeps the degenerate "repeated open tag,
        # no JSON" case linear: raw_decode is never invoked (its
        # JSONDecodeError construction scans the whole prefix, which would be
        # quadratic over many failing openers).
        if idx >= n or text[idx] != "{":
            pos = start + len(open_tag)
            continue
        try:
            payload, end = decoder.raw_decode(text, idx)
        except (json.JSONDecodeError, ValueError):
            # Malformed JSON right after this open tag; skip past the tag
            # and keep scanning for the next opener.
            pos = start + len(open_tag)
            continue
        if isinstance(payload, dict):
            name = payload.get("name") or payload.get("function") or payload.get("tool")
            args = payload.get("arguments") or payload.get("input") or payload.get("parameters") or {}
            if not isinstance(args, dict):
                args = {}
            if name:
                out.append(ParsedToolCall(name=str(name), arguments=args))
        # Resume scanning after the consumed JSON value, stepping past the
        # close tag when it immediately follows (allowing intervening space).
        pos = end
        skip = end
        while skip < n and text[skip].isspace():
            skip += 1
        if text.startswith(close_tag, skip):
            pos = skip + len(close_tag)
    return out


class XmlToolCallParser(ToolCallParser):
    """Parser for the ``<tool_call>...</tool_call>`` XML-style tag format.

    This is the form emitted by several open-source model families that
    follow the ``<tool_call>...</tool_call>`` XML convention (notably a
    number of Mistral fine-tunes and Qwen-Agent derivatives).
    """

    name = "xml_tool_call"

    def parse(self, text: str) -> list[ParsedToolCall]:
        """Extract tool calls delimited by ``<tool_call>...</tool_call>``."""
        return parse_tool_call_blocks(text, open_tag="<tool_call>", close_tag="</tool_call>")


class LlamaParser(ToolCallParser):
    """Parser for Llama 3.x tool-call formats.

    Recognises both ``<|python_tag|>{json}<|eom_id|>`` and the alternate
    ``<function=name>{json}</function>`` shape some Llama-derivative chat
    templates emit.
    """

    name = "llama"

    def parse(self, text: str) -> list[ParsedToolCall]:
        """Extract Llama tool calls in either supported shape."""
        out: list[ParsedToolCall] = []
        for m in re.finditer(r"<\|python_tag\|>(.*?)<\|eom_id\|>", text, re.S):
            payload = _safe_json(m.group(1))
            if not payload:
                continue
            name = payload.get("name", "")
            params = payload.get("parameters") or payload.get("arguments") or {}
            if not isinstance(params, dict):
                params = {}
            if name:
                out.append(ParsedToolCall(name=name, arguments=params))
        for m in re.finditer(r"<function=([^>]+)>(.*?)</function>", text, re.S):
            args = _safe_json(m.group(2)) or {}
            out.append(ParsedToolCall(name=m.group(1).strip(), arguments=args))
        return out
