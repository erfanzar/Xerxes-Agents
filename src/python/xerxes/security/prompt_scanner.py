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
and unicode checks; matched content is replaced with a ``[BLOCKED:
...]`` placeholder so the model never sees the payload. The checks
are cheap, deterministic and best-effort: they catch unsophisticated
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
    """Return ``content`` unchanged, or a ``[BLOCKED: ...]`` placeholder.

    Runs every compiled threat pattern and checks for known invisible
    unicode codepoints. If any rule matches, the original content is
    discarded and replaced with a short marker naming the matched
    pattern ids. ``filename`` only appears in log messages and the
    placeholder string."""

    findings: list[str] = []

    for char in _CONTEXT_INVISIBLE_CHARS:
        if char in content:
            findings.append(f"invisible_unicode_U+{ord(char):04X}")

    for compiled, pid in _COMPILED_PATTERNS:
        if compiled.search(content):
            findings.append(pid)

    if not findings:
        return content

    logger.warning("Context file %s blocked: %s", filename, ", ".join(findings))
    return f"[BLOCKED: {filename} contained potential prompt injection ({', '.join(findings)}). Content not loaded.]"


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
