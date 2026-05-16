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
"""Agent-facing tools for reading and updating the markdown workspace.

Wraps :class:`MarkdownAgentWorkspace` so the model can list, read, write,
append, and diff workspace files. Every public function resolves the supplied
relative path against the workspace root and raises ``ValueError`` if the
result escapes — the agent cannot reach arbitrary disk locations through these
tools.
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Any

from ..channels.workspace import MarkdownAgentWorkspace


def _resolve_inside(workspace_root: Path, relative_path: str) -> Path:
    """Resolve ``relative_path`` ensuring it stays inside ``workspace_root``.

    Raises ``ValueError`` for any ``..`` escape or absolute paths."""
    if not relative_path:
        raise ValueError("workspace path must be non-empty")
    candidate = (workspace_root / relative_path).resolve()
    root = workspace_root.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise ValueError(f"path {relative_path!r} escapes workspace root") from e
    return candidate


def workspace_list(workspace: MarkdownAgentWorkspace | None = None) -> list[dict[str, Any]]:
    """Return metadata for every file in the workspace.

    Each entry is a dict with ``path`` (relative), ``bytes``, and
    ``modified`` (epoch seconds)."""

    ws = workspace or MarkdownAgentWorkspace()
    ws.ensure()
    root = ws.path.resolve()
    out: list[dict[str, Any]] = []
    for current_dir, _, files in os.walk(root):
        for name in files:
            full = Path(current_dir) / name
            rel = full.relative_to(root)
            try:
                stat = full.stat()
            except OSError:
                continue
            out.append({"path": str(rel), "bytes": stat.st_size, "modified": stat.st_mtime})
    return sorted(out, key=lambda d: d["path"])


def workspace_read(
    path: str,
    workspace: MarkdownAgentWorkspace | None = None,
    *,
    max_bytes: int | None = None,
) -> str:
    """Read a workspace file as text.

    Raises ``FileNotFoundError`` if the file is missing and
    ``ValueError`` if ``path`` escapes the workspace root."""

    ws = workspace or MarkdownAgentWorkspace()
    ws.ensure()
    target = _resolve_inside(ws.path, path)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"workspace file not found: {path}")
    text = target.read_text(encoding="utf-8")
    if max_bytes is not None and len(text) > max_bytes:
        return text[:max_bytes] + f"\n[... truncated; {len(text) - max_bytes} bytes elided]"
    return text


def workspace_write(
    path: str,
    content: str,
    workspace: MarkdownAgentWorkspace | None = None,
    *,
    create_dirs: bool = True,
) -> dict[str, Any]:
    """Overwrite a workspace file. Creates parent dirs by default.

    Returns a result dict: ``{path, bytes, created}``."""

    ws = workspace or MarkdownAgentWorkspace()
    ws.ensure()
    target = _resolve_inside(ws.path, path)
    created = not target.exists()
    if create_dirs:
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"path": str(target.relative_to(ws.path.resolve())), "bytes": len(content), "created": created}


def workspace_append(
    path: str,
    content: str,
    workspace: MarkdownAgentWorkspace | None = None,
    *,
    create_dirs: bool = True,
    ensure_newline: bool = True,
) -> dict[str, Any]:
    """Append text to a workspace file. Creates it if missing.

    By default, ensures there's a newline between existing content and
    the new content. Returns ``{path, appended_bytes, created}``."""

    ws = workspace or MarkdownAgentWorkspace()
    ws.ensure()
    target = _resolve_inside(ws.path, path)
    created = not target.exists()
    if create_dirs:
        target.parent.mkdir(parents=True, exist_ok=True)
    prefix = ""
    if not created and ensure_newline:
        existing = target.read_text(encoding="utf-8")
        if existing and not existing.endswith("\n"):
            prefix = "\n"
    with target.open("a", encoding="utf-8") as fh:
        fh.write(prefix + content)
    return {
        "path": str(target.relative_to(ws.path.resolve())),
        "appended_bytes": len(prefix) + len(content),
        "created": created,
    }


def workspace_diff(
    path: str,
    new_content: str,
    workspace: MarkdownAgentWorkspace | None = None,
) -> str:
    """Return a unified diff between the current file content and ``new_content``.

    If the file does not exist, the diff is against an empty file."""

    ws = workspace or MarkdownAgentWorkspace()
    ws.ensure()
    target = _resolve_inside(ws.path, path)
    if target.exists() and target.is_file():
        before = target.read_text(encoding="utf-8")
    else:
        before = ""
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=3,
    )
    return "".join(diff)


__all__ = [
    "workspace_append",
    "workspace_diff",
    "workspace_list",
    "workspace_read",
    "workspace_write",
]
