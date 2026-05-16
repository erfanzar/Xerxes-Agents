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
"""Line-oriented CRUD for the workspace ``MEMORY.md`` and ``USER.md`` files.

Exposes eight functions: ``memory_add``,
``memory_list``, ``memory_replace``, ``memory_remove`` operate on the
workspace ``MEMORY.md``; the matching ``user_*`` quartet operates on
``USER.md``. Both files are simple newline-delimited markdown with blank
lines stripped; the 1-based ``id`` returned by these tools is the line index
within the file at the time of the call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _read_lines(path: Path) -> list[str]:
    """Return non-blank lines from ``path``, or an empty list when it is missing."""
    if not path.exists():
        return []
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_lines(path: Path, lines: list[str]) -> None:
    """Persist ``lines`` to ``path`` joined by newlines, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _add(path: Path, content: str) -> dict[str, Any]:
    """Append a stripped ``content`` line and return its 1-based id."""
    if not content.strip():
        return {"ok": False, "error": "content required"}
    lines = _read_lines(path)
    lines.append(content.strip())
    _write_lines(path, lines)
    return {"ok": True, "id": len(lines), "content": content.strip()}


def _list(path: Path, *, limit: int | None = None) -> dict[str, Any]:
    """Return every line with its 1-based id, optionally tail-truncated."""
    lines = _read_lines(path)
    items = [{"id": i + 1, "content": line} for i, line in enumerate(lines)]
    if limit is not None:
        items = items[-int(limit) :]
    return {"ok": True, "items": items}


def _replace(path: Path, entry_id: int, content: str) -> dict[str, Any]:
    """Overwrite the line at ``entry_id`` with the stripped ``content``."""
    if not content.strip():
        return {"ok": False, "error": "content required"}
    lines = _read_lines(path)
    if not 1 <= entry_id <= len(lines):
        return {"ok": False, "error": f"id {entry_id} not found"}
    lines[entry_id - 1] = content.strip()
    _write_lines(path, lines)
    return {"ok": True, "id": entry_id, "content": content.strip()}


def _remove(path: Path, entry_id: int) -> dict[str, Any]:
    """Delete the line at ``entry_id`` and return the removed content."""
    lines = _read_lines(path)
    if not 1 <= entry_id <= len(lines):
        return {"ok": False, "error": f"id {entry_id} not found"}
    removed = lines.pop(entry_id - 1)
    _write_lines(path, lines)
    return {"ok": True, "id": entry_id, "content": removed}


# ---------------------------- public surface -------------------------------


def memory_add(workspace_path: Path, content: str) -> dict[str, Any]:
    """Append ``content`` to ``MEMORY.md`` inside ``workspace_path``."""
    return _add(workspace_path / "MEMORY.md", content)


def memory_list(workspace_path: Path, *, limit: int | None = None) -> dict[str, Any]:
    """Return ``MEMORY.md`` entries, optionally limited to the last ``limit`` lines."""
    return _list(workspace_path / "MEMORY.md", limit=limit)


def memory_replace(workspace_path: Path, entry_id: int, content: str) -> dict[str, Any]:
    """Replace the ``entry_id``-th line of ``MEMORY.md`` with ``content``."""
    return _replace(workspace_path / "MEMORY.md", entry_id, content)


def memory_remove(workspace_path: Path, entry_id: int) -> dict[str, Any]:
    """Delete the ``entry_id``-th line of ``MEMORY.md``."""
    return _remove(workspace_path / "MEMORY.md", entry_id)


def user_add(workspace_path: Path, content: str) -> dict[str, Any]:
    """Append ``content`` to ``USER.md`` inside ``workspace_path``."""
    return _add(workspace_path / "USER.md", content)


def user_list(workspace_path: Path, *, limit: int | None = None) -> dict[str, Any]:
    """Return ``USER.md`` entries, optionally limited to the last ``limit`` lines."""
    return _list(workspace_path / "USER.md", limit=limit)


def user_replace(workspace_path: Path, entry_id: int, content: str) -> dict[str, Any]:
    """Replace the ``entry_id``-th line of ``USER.md`` with ``content``."""
    return _replace(workspace_path / "USER.md", entry_id, content)


def user_remove(workspace_path: Path, entry_id: int) -> dict[str, Any]:
    """Delete the ``entry_id``-th line of ``USER.md``."""
    return _remove(workspace_path / "USER.md", entry_id)


__all__ = [
    "memory_add",
    "memory_list",
    "memory_remove",
    "memory_replace",
    "user_add",
    "user_list",
    "user_remove",
    "user_replace",
]
