# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.


"""Permission modes for tool execution in the streaming agent loop.

Inspired by the nano-claude-code permission gate, this module implements a
three-mode permission system that determines whether a tool invocation
should be auto-approved or requires user confirmation.

Modes:

- ``auto`` (default): Read-only and safe tools are auto-approved.
  Write operations (Write, Edit), destructive Bash commands, and
  unknown tools require user permission.
- ``accept-all``: All tools are auto-approved without asking.
- ``manual``: Every tool invocation requires explicit user permission.

The :func:`check_permission` function is the main entry point. It takes
a tool call dict and returns ``True`` if auto-approved.

Usage::

    from xerxes.streaming.permissions import PermissionMode, check_permission


    assert check_permission({"name": "Read", "input": {}}, PermissionMode.AUTO)
    assert not check_permission({"name": "Edit", "input": {}}, PermissionMode.AUTO)


    assert check_permission({"name": "Edit", "input": {}}, PermissionMode.ACCEPT_ALL)
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any


class PermissionMode(Enum):
    """Permission mode controlling tool execution approval.

    Attributes:
        AUTO: Smart auto-approve for safe tools, ask for writes/destructive ops.
        ACCEPT_ALL: Approve everything without asking.
        MANUAL: Always ask for permission.
    """

    AUTO = "auto"
    ACCEPT_ALL = "accept-all"
    MANUAL = "manual"


SAFE_TOOLS: frozenset[str] = frozenset(
    {
        "ReadFile",
        "GlobTool",
        "GrepTool",
        "ListDir",
        "WebScraper",
        "APIClient",
        "RSSReader",
        "URLAnalyzer",
        "GoogleSearch",
        "DuckDuckGoSearch",
        "SystemInfo",
        "skills_list",
        "skill_view",
        "session_search",
        "search_memory",
        "get_memory_statistics",
        "consolidate_agent_memories",
        "TaskListTool",
        "TaskGetTool",
        "TaskOutputTool",
        "ToolSearchTool",
        "JSONProcessor",
        "CSVProcessor",
        "TextProcessor",
        "Calculator",
        "StatisticalAnalyzer",
        "MathematicalFunctions",
        "UnitConverter",
        "DateTimeProcessor",
        "URLAnalyzer",
    }
)

_SAFE_BASH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*(ls|pwd|whoami|date|uname|cat|head|tail|wc|file|which|type|echo)\b"),
    re.compile(r"^\s*git\s+(status|log|diff|branch|show|remote|tag|stash\s+list)\b"),
    re.compile(r"^\s*(find|grep|rg|fd|ag|ack|tree)\b"),
    re.compile(r"^\s*(python|python3|node|ruby|go|cargo|rustc)\s+--version\b"),
    re.compile(r"^\s*(npm|yarn|pnpm|pip|pip3|cargo|go)\s+(list|show|info|search|outdated)\b"),
    re.compile(r"^\s*(env|printenv|hostname|id|groups|locale|df|du|free|uptime|top\s+-l\s*1)\b"),
]

_DANGEROUS_BASH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+(-[a-zA-Z]*f|-[a-zA-Z]*r|--force|--recursive)\b"),
    re.compile(r"\bgit\s+(push\s+--force|reset\s+--hard|clean\s+-[a-zA-Z]*f)\b"),
    re.compile(r"\b(mkfs|dd\s+if=|format|fdisk|parted)\b"),
    re.compile(r">\s*/dev/"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bcurl\b.*\|\s*(bash|sh|zsh)\b"),
]


def is_safe_bash(command: str) -> bool:
    """Check if a Bash command is safe to auto-approve.

    A command is safe if it matches at least one safe pattern and
    does not match any dangerous pattern.

    Args:
        command: The shell command string.

    Returns:
        ``True`` if the command is considered safe.
    """
    for pattern in _DANGEROUS_BASH_PATTERNS:
        if pattern.search(command):
            return False

    for pattern in _SAFE_BASH_PATTERNS:
        if pattern.search(command):
            return True

    return False


def check_permission(
    tool_call: dict[str, Any],
    mode: PermissionMode = PermissionMode.AUTO,
) -> bool:
    """Check if a tool invocation is auto-approved under the given mode.

    Args:
        tool_call: Dict with ``"name"`` and ``"input"`` keys.
        mode: The active permission mode.

    Returns:
        ``True`` if the tool call is auto-approved (no user prompt needed).
        ``False`` if user permission is required.
    """
    if mode == PermissionMode.ACCEPT_ALL:
        return True
    if mode == PermissionMode.MANUAL:
        return False

    name = tool_call.get("name", "")

    if name in SAFE_TOOLS:
        return True

    if name in ("Bash", "ExecuteShell"):
        cmd = tool_call.get("input", {}).get("command", "")
        return is_safe_bash(cmd)

    if name == "ExecutePythonCode":
        code = tool_call.get("input", {}).get("code", "")
        if re.search(r"\b(open\(.*['\"]w|subprocess|os\.system|os\.popen|shutil)\b", code):
            return False
        return True

    if name in ("Agent", "SendMessage"):
        return True

    if name in ("MemorySave", "MemoryDelete"):
        return True

    if name in ("Write", "WriteFile", "Edit", "FileEditTool", "AppendFile"):
        return False

    return False


def format_permission_description(tool_call: dict[str, Any]) -> str:
    """Generate a human-readable description for a permission request.

    Args:
        tool_call: Dict with ``"name"`` and ``"input"`` keys.

    Returns:
        A short description string suitable for display.
    """
    name = tool_call.get("name", "")
    inp = tool_call.get("input", {})

    if name in ("Bash", "ExecuteShell"):
        return f"Run: {inp.get('command', '')}"
    if name in ("Write", "WriteFile"):
        return f"Write to: {inp.get('file_path', '')}"
    if name in ("Edit",):
        return f"Edit: {inp.get('file_path', '')}"
    if name == "AppendFile":
        return f"Append to: {inp.get('file_path', '')}"
    if name == "ExecutePythonCode":
        code = inp.get("code", "")
        preview = code[:80].replace("\n", " ")
        return f"Execute Python: {preview}..."

    first_val = next(iter(inp.values()), "") if inp else ""
    preview = str(first_val)[:60]
    return f"{name}({preview})"


__all__ = [
    "SAFE_TOOLS",
    "PermissionMode",
    "check_permission",
    "format_permission_description",
    "is_safe_bash",
]
