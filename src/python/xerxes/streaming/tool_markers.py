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
"""Helpers for provider-only tool-call markers that must never reach UI text."""

from __future__ import annotations

import json
import re
from html import unescape
from typing import Any

ASSISTANT_TOOL_CALLS_MARKER = "ASSISTANT_TOOL_CALLS"
_INVOKE_BLOCK_RE = re.compile(
    r"<invoke\s+name=(?P<quote>[\"'])(?P<name>.*?)(?P=quote)[^>]*>(?P<body>.*?)</invoke>",
    re.S,
)
_PARAMETER_BLOCK_RE = re.compile(
    r"<parameter\s+name=(?P<quote>[\"'])(?P<name>.*?)(?P=quote)[^>]*>(?P<body>.*?)</parameter>",
    re.S,
)
_SYSTEM_REMINDER_RE = re.compile(
    r"[ \t]*<system-reminder\b[^>]*>.*?</system-reminder>[ \t]*(?:\n)?",
    re.S,
)
_TOOL_CALL_ID_LINE_RE = re.compile(r"(?m)^TOOL_CALL_ID:\s*[^\n]*(?:\n|$)")
_TOOL_LINE_RE = re.compile(r"(?m)^TOOL:\s*(?:\{|\[|None\b|True\b|False\b|\"|')[^\n]*(?:\n|$)")


def extract_assistant_tool_call_markers(
    text: str, *, id_prefix: str = "call_marker"
) -> tuple[str, list[dict[str, Any]]]:
    """Strip assistant tool-call marker blocks and return normalized tool calls."""
    decoder = json.JSONDecoder()
    calls: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []

    if ASSISTANT_TOOL_CALLS_MARKER in text:
        search_from = 0
        while True:
            marker_start = text.find(ASSISTANT_TOOL_CALLS_MARKER, search_from)
            if marker_start < 0:
                break
            payload_start = marker_start + len(ASSISTANT_TOOL_CALLS_MARKER)
            while payload_start < len(text) and text[payload_start].isspace():
                payload_start += 1
            if payload_start < len(text) and text[payload_start] == ":":
                payload_start += 1
            while payload_start < len(text) and text[payload_start].isspace():
                payload_start += 1
            try:
                payload, consumed = decoder.raw_decode(text[payload_start:])
            except json.JSONDecodeError:
                search_from = payload_start
                continue

            marker_calls = _normalize_marker_payload(payload, id_prefix=id_prefix, start_index=len(calls))
            if marker_calls:
                calls.extend(marker_calls)
                spans.append((marker_start, payload_start + consumed))
            search_from = payload_start + consumed

    for match in _INVOKE_BLOCK_RE.finditer(text):
        tool_call = _normalize_invoke_block(
            match.group("name"),
            match.group("body"),
            fallback_id=f"{id_prefix}_{len(calls)}",
        )
        if tool_call is not None:
            calls.append(tool_call)
            spans.append(match.span())

    clean = _remove_spans(text, spans) if spans else text
    return _strip_provider_tool_context(clean).strip(), calls


def strip_assistant_tool_call_markers(text: str) -> str:
    """Remove provider-only assistant tool-call marker blocks from text."""
    clean, _ = extract_assistant_tool_call_markers(text)
    return clean


def _strip_provider_tool_context(text: str) -> str:
    clean = _SYSTEM_REMINDER_RE.sub("", text)
    clean = _TOOL_CALL_ID_LINE_RE.sub("", clean)
    clean = _strip_json_line_marker(clean, "TOOL:")
    clean = _TOOL_LINE_RE.sub("", clean)
    return clean.strip()


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    clean = text
    for start, end in sorted(spans, reverse=True):
        before = clean[:start].rstrip()
        after = clean[end:].lstrip()
        clean = f"{before}\n{after}" if before and after else before or after
    return clean


def _strip_json_line_marker(text: str, marker: str) -> str:
    decoder = json.JSONDecoder()
    spans: list[tuple[int, int]] = []
    for match in re.finditer(rf"(?m)^{re.escape(marker)}\s*", text):
        payload_start = match.end()
        try:
            _, consumed = decoder.raw_decode(text[payload_start:])
        except json.JSONDecodeError:
            continue
        spans.append((match.start(), payload_start + consumed))

    clean = text
    for start, end in reversed(spans):
        before = clean[:start].rstrip()
        after = clean[end:].lstrip()
        clean = f"{before}\n{after}" if before and after else before or after
    return clean


def _normalize_marker_payload(payload: Any, *, id_prefix: str, start_index: int) -> list[dict[str, Any]]:
    items = payload if isinstance(payload, list) else [payload]
    calls: list[dict[str, Any]] = []
    for offset, item in enumerate(items):
        call = _normalize_marker_call(item, fallback_id=f"{id_prefix}_{start_index + offset}")
        if call is not None:
            calls.append(call)
    return calls


def _normalize_invoke_block(name: str, body: str, *, fallback_id: str) -> dict[str, Any] | None:
    name = name.strip()
    if not name:
        return None

    args: dict[str, Any] = {}
    for match in _PARAMETER_BLOCK_RE.finditer(body):
        key = match.group("name").strip()
        if key:
            args[key] = _decode_parameter_value(match.group("body"))

    return {"id": fallback_id, "name": name, "input": args}


def _decode_parameter_value(value: str) -> Any:
    clean = unescape(value).strip()
    if not clean:
        return ""
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return clean


def _normalize_marker_call(item: Any, *, fallback_id: str) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    function = item.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        raw_input = function.get("arguments")
    else:
        name = item.get("name") or item.get("tool_name")
        raw_input = item.get("input", item.get("arguments", {}))

    if not name:
        return None

    if isinstance(raw_input, str):
        try:
            parsed = json.loads(raw_input)
        except json.JSONDecodeError:
            parsed = {}
    elif isinstance(raw_input, dict):
        parsed = raw_input
    else:
        parsed = {}

    return {
        "id": str(item.get("id") or item.get("tool_call_id") or fallback_id),
        "name": str(name),
        "input": parsed,
    }
