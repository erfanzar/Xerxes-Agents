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
"""Agent-facing wrappers around ``runtime.agent_memory``.

Six tools the agent calls during conversation:

    * ``agent_memory_read(scope, path)``         — read a memory file.
    * ``agent_memory_write(scope, path, body)``  — overwrite.
    * ``agent_memory_append(scope, path, body)`` — append (with optional section/timestamp).
    * ``agent_memory_list(scope)``               — list files.
    * ``agent_memory_search(query, scope)``      — substring-search across files.
    * ``agent_memory_journal(scope, note)``      — quick timestamped line for today.

The shared ``AgentMemory`` instance is held in a module-global so the
daemon can install it once per session and every tool call sees the
same configuration."""

from __future__ import annotations

from typing import Any

from ..runtime.agent_memory import AgentMemory

_current: AgentMemory | None = None


def set_active_memory(memory: AgentMemory | None) -> None:
    """Install (or clear) the active ``AgentMemory`` instance.

    Called by the daemon when a session is opened. ``None`` means
    no memory is configured for the current session (tools return an
    error in that case)."""
    global _current
    _current = memory


def active_memory() -> AgentMemory | None:
    """Return the currently installed :class:`AgentMemory`, or ``None``."""
    return _current


_NOT_CONFIGURED = {"ok": False, "error": "agent memory not configured for this session"}


# ---------------------------- tool surface ---------------------------------


def agent_memory_read(scope: str, path: str) -> dict[str, Any]:
    """Read a memory file. ``scope`` is ``"global"`` or ``"project"``."""
    if _current is None:
        return dict(_NOT_CONFIGURED)
    try:
        text = _current.read(scope, path)
    except (FileNotFoundError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "scope": scope, "path": path, "body": text, "bytes": len(text)}


def agent_memory_write(scope: str, path: str, body: str) -> dict[str, Any]:
    """Overwrite ``path`` inside the given scope."""
    if _current is None:
        return dict(_NOT_CONFIGURED)
    try:
        result = _current.write(scope, path, body)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, **result}


def agent_memory_append(
    scope: str,
    path: str,
    body: str,
    *,
    section: str = "",
    timestamp: bool = True,
) -> dict[str, Any]:
    """Append to ``path``. Optionally wrap with a ``## section`` header and a UTC timestamp."""
    if _current is None:
        return dict(_NOT_CONFIGURED)
    try:
        result = _current.append(scope, path, body, section=section, timestamp=timestamp)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, **result}


def agent_memory_list(scope: str | None = None) -> dict[str, Any]:
    """List every memory file in one or both scopes."""
    if _current is None:
        return dict(_NOT_CONFIGURED)
    files = _current.list_files(scope)
    return {"ok": True, "scope": scope or "all", "files": [f.to_dict() for f in files], "count": len(files)}


def agent_memory_search(query: str, scope: str | None = None, *, limit: int = 20) -> dict[str, Any]:
    """Substring-search across memory files."""
    if _current is None:
        return dict(_NOT_CONFIGURED)
    hits = _current.search(query, scope=scope, limit=limit)
    return {"ok": True, "query": query, "hits": hits, "count": len(hits)}


def agent_memory_journal(scope: str, note: str) -> dict[str, Any]:
    """Append a timestamped note to today's journal file."""
    if _current is None:
        return dict(_NOT_CONFIGURED)
    try:
        result = _current.journal(scope, note)
    except (FileNotFoundError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, **result}


def agent_memory_status() -> dict[str, Any]:
    """Report what memory the agent has access to (paths + file counts)."""
    mem = _current
    if mem is None:
        return {"ok": True, "available": False}
    status = mem.status()
    return {"ok": True, "available": True, **status}


__all__ = [
    "active_memory",
    "agent_memory_append",
    "agent_memory_journal",
    "agent_memory_list",
    "agent_memory_read",
    "agent_memory_search",
    "agent_memory_status",
    "agent_memory_write",
    "set_active_memory",
]
