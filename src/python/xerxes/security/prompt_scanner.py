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
"""Static prompt-injection scanner for context files.

When the user (or a tool) pulls external content into the model's
context, that content can carry adversarial instructions —
"ignore previous instructions", base64-wrapped curl payloads, hidden
HTML, zero-width characters. This module runs a small set of regex
and unicode checks; each matched span (and each invisible codepoint)
is replaced in place with a short ``[BLOCKED: ...]`` placeholder while
the surrounding legitimate text is preserved, so the model never sees
the payload but does keep the rest of the message. The checks are
cheap, deterministic and best-effort: they catch unsophisticated
injections, not a targeted attacker."""

from __future__ import annotations

import logging
import os
import re
from typing import Final

logger = logging.getLogger(__name__)

_CONTEXT_THREAT_PATTERNS: Final[list[tuple[str, str]]] = [
    (r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions", "prompt_injection"),
    (r"do\s+not\s+tell\s+the\s+user", "deception_hide"),
    (r"system\s+prompt\s+override", "sys_prompt_override"),
    (r"disregard\s+(your\s+)?(all\s+)?(any\s+)?(instructions|rules|guidelines)", "disregard_rules"),
    (
        r"act\s+as\s+(if|though)\s+you\s+(have\s+no|don\'t\s+have)\s+(any\s+)?(restrictions|limits|rules)",
        "bypass_restrictions",
    ),
    (
        r"<!--[^>]*(?:ignore|override|system|secret|hidden)[^>]*-->",
        "html_comment_injection",
    ),
    (r"<\s*div\s+style\s*=\s*[\"\'][\s\S]*?display\s*:\s*none", "hidden_div"),
    (
        r"translate\s+.*\s+into\s+.*\s+and\s+(execute|run|eval)",
        "translate_execute",
    ),
    (
        r"curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)",
        "exfil_curl",
    ),
    (
        r"cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass)",
        "read_secrets",
    ),
]

_COMPILED_PATTERNS: Final[list[tuple[re.Pattern[str], str]]] = [
    (re.compile(pat, re.IGNORECASE), pid) for pat, pid in _CONTEXT_THREAT_PATTERNS
]

_CONTEXT_INVISIBLE_CHARS: Final[set[str]] = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\ufeff",
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
}


def scan_context_content(content: str, filename: str = "unknown") -> str:
    """Return ``content`` with each detected threat span neutralised in place.

    Runs every compiled threat pattern and checks for known invisible
    unicode codepoints. Each matched span (and each invisible codepoint)
    is replaced with a short ``[BLOCKED: <filename> <pid>]`` placeholder
    while the surrounding legitimate text is preserved, so a message that
    merely quotes a flagged phrase keeps the rest of its body instead of
    being discarded wholesale. Clean content is returned unchanged.
    ``filename`` appears in log messages and in the placeholders."""

    # Collect every threatening span as (start, end, pid) over the original
    # text. Invisible codepoints are single-character spans; regex matches use
    # their reported span. We scan the original (unmodified) text for all
    # patterns so that an inserted placeholder can never itself be re-matched.
    spans: list[tuple[int, int, str]] = []
    findings: list[str] = []

    for idx, char in enumerate(content):
        if char in _CONTEXT_INVISIBLE_CHARS:
            pid = f"invisible_unicode_U+{ord(char):04X}"
            spans.append((idx, idx + 1, pid))
            findings.append(pid)

    for compiled, pid in _COMPILED_PATTERNS:
        for m in compiled.finditer(content):
            spans.append((m.start(), m.end(), pid))
            findings.append(pid)

    if not spans:
        return content

    # Merge overlapping/adjacent spans (regex patterns may overlap, and an
    # invisible char may sit inside a phrase match) into a single placeholder
    # so the output stays readable. Sort by start, then by widest end.
    spans.sort(key=lambda s: (s[0], -s[1]))
    merged: list[tuple[int, int, list[str]]] = []
    for start, end, pid in spans:
        if merged and start <= merged[-1][1]:
            prev_start, prev_end, pids = merged[-1]
            if pid not in pids:
                pids.append(pid)
            merged[-1] = (prev_start, max(prev_end, end), pids)
        else:
            merged.append((start, end, [pid]))

    parts: list[str] = []
    cursor = 0
    for start, end, pids in merged:
        parts.append(content[cursor:start])
        parts.append(f"[BLOCKED: {filename} {', '.join(pids)}]")
        cursor = end
    parts.append(content[cursor:])

    logger.warning("Context file %s neutralised: %s", filename, ", ".join(findings))
    return "".join(parts)


def scan_context_file(path: str | os.PathLike[str], filename: str | None = None) -> str:
    """Read ``path`` as UTF-8 and pass through :func:`scan_context_content`.

    Unreadable files produce a ``[BLOCKED: ... unreadable ...]`` placeholder
    rather than propagating the IO error, so a single bad file cannot
    cancel the surrounding context-load pipeline."""

    from pathlib import Path

    p = Path(path)
    name = filename or p.name
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not read context file %s: %s", name, exc)
        return f"[BLOCKED: {name} unreadable ({exc})]"
    return scan_context_content(text, filename=name)
