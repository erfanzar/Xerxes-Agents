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
"""Prompt scanner module for Xerxes.

Exports:
    - logger
    - scan_context_content
    - scan_context_file"""

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
    """Scan context content.

    Args:
        content (str): IN: content. OUT: Consumed during execution.
        filename (str, optional): IN: filename. Defaults to 'unknown'. OUT: Consumed during execution.
    Returns:
        str: OUT: Result of the operation."""

    findings: list[str] = []
    """Scan context content.

    Args:
        content (str): IN: content. OUT: Consumed during execution.
        filename (str, optional): IN: filename. Defaults to 'unknown'. OUT: Consumed during execution.
    Returns:
        str: OUT: Result of the operation."""
    """Scan context content.

    Args:
        content (str): IN: content. OUT: Consumed during execution.
        filename (str, optional): IN: filename. Defaults to 'unknown'. OUT: Consumed during execution.
    Returns:
        str: OUT: Result of the operation."""

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
    """Scan context file.

    Args:
        path (str | os.PathLike[str]): IN: path. OUT: Consumed during execution.
        filename (str | None, optional): IN: filename. Defaults to None. OUT: Consumed during execution.
    Returns:
        str: OUT: Result of the operation."""

    from pathlib import Path

    """Scan context file.

    Args:
        path (str | os.PathLike[str]): IN: path. OUT: Consumed during execution.
        filename (str | None, optional): IN: filename. Defaults to None. OUT: Consumed during execution.
    Returns:
        str: OUT: Result of the operation."""
    """Scan context file.

    Args:
        path (str | os.PathLike[str]): IN: path. OUT: Consumed during execution.
        filename (str | None, optional): IN: filename. Defaults to None. OUT: Consumed during execution.
    Returns:
        str: OUT: Result of the operation."""

    p = Path(path)
    name = filename or p.name
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not read context file %s: %s", name, exc)
        return f"[BLOCKED: {name} unreadable ({exc})]"
    return scan_context_content(text, filename=name)
