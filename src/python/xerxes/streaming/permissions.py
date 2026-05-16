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
"""Permission gating for the streaming loop.

Every tool call passes through :func:`check_permission` before execution. The
gate consults a hand-curated allowlist (:data:`SAFE_TOOLS`), pattern-matches
shell commands against safe/dangerous regex sets, and applies extra rules
for :func:`ExecutePythonCode` and any destructive file-writing tools.

The mode (:class:`PermissionMode`) decides whether the gate is a hard answer
(``ACCEPT_ALL`` / ``MANUAL``) or merely a fast-path approval for known-safe
calls (``AUTO``), with anything else escalated to the user.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any


class PermissionMode(Enum):
    """Permission-gate behaviour selector.

    Attributes:
        AUTO: Auto-approve known-safe tools; prompt for everything else.
        ACCEPT_ALL: Approve every tool call (use only in sandboxed runs).
        MANUAL: Deny every tool call until the user explicitly approves.
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
        "AskUserQuestionTool",
        "SetInteractionModeTool",
        "JSONProcessor",
        "CSVProcessor",
        "TextProcessor",
        "Calculator",
        "StatisticalAnalyzer",
        "MathematicalFunctions",
        "UnitConverter",
        "DateTimeProcessor",
    }
)

_SAFE_BASH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*(ls|pwd|whoami|date|uname|cat|head|tail|wc|file|which|type|echo)\b"),
    re.compile(r"^\s*cd(?:\s+(?:--\s+)?[^\n;&|`$()<>]+)?\s*(?:&&\s*pwd\s*)?$"),
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
    """Return whether a shell command is on the conservative auto-approve list.

    The check rejects dangerous patterns first (``rm -rf``, ``sudo``,
    ``curl ... | sh``, partition tools, ``git push --force``) and then accepts
    only commands whose first token matches a read-only allowlist (``ls``,
    ``git status``, search tools, version probes, etc.). A ``cd ... &&``
    prefix is unwrapped and the trailing command re-evaluated recursively.

    Args:
        command: Raw shell command string from the tool call.

    Returns:
        ``True`` only if every check passes; ``False`` is the safe default.
    """

    command = command.strip()

    cd_prefix = re.match(r"^cd(?:\s+(?:--\s+)?[^\n;&|`$()<>]+)?\s*&&\s*(.+)$", command)
    if cd_prefix:
        return is_safe_bash(cd_prefix.group(1))

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
    """Decide whether a tool call may proceed without user prompting.

    In ``AUTO`` mode the gate auto-approves any tool in :data:`SAFE_TOOLS`,
    delegates bash/shell calls to :func:`is_safe_bash`, blocks Python code
    that touches the filesystem or shells out, and refuses all file-writing
    tools so they always escalate. ``ACCEPT_ALL`` and ``MANUAL`` short-circuit
    to ``True`` / ``False`` respectively.

    Args:
        tool_call: ``{"name", "input"}`` dict as yielded by the loop.
        mode: Active permission mode.

    Returns:
        ``True`` to skip prompting and execute immediately, ``False`` to
        emit a :class:`PermissionRequest`.
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
    """Render a one-line description of a tool call for the approval UI.

    Shell tools display the command, write/edit tools display the target
    file path, ExecutePythonCode shows a code preview, and generic tools fall
    back to ``Name(first-arg)``.
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
