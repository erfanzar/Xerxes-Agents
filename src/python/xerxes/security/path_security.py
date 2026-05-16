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
"""Path security — block ``..`` escapes out of the working directory.

Used by every file tool before opening a path supplied by the model.
The pure ``resolve_within`` helper raises ``PathEscape`` when the
resolved path would leave the workspace; ``safe_path`` returns
``None`` instead for callers preferring soft denial.

Exports:
    - PathEscape
    - resolve_within
    - safe_path"""

from __future__ import annotations

from pathlib import Path


class PathEscape(ValueError):
    """Raised when a path resolves outside the allowed workspace."""


def resolve_within(workspace: str | Path, candidate: str | Path) -> Path:
    """Resolve ``candidate`` relative to ``workspace`` and confirm it stays inside.

    Symlinks are followed during resolution; the final path must still
    be a descendant of ``workspace`` after that. Absolute candidates
    are interpreted relative to the workspace, never as system paths."""
    root = Path(workspace).expanduser().resolve()
    target = Path(candidate)
    if target.is_absolute():
        # Force absolute paths to be re-rooted under workspace.
        target = Path(*target.parts[1:]) if target.parts else target
    resolved = (root / target).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as e:
        raise PathEscape(f"path {candidate!r} escapes workspace root {root}") from e
    return resolved


def safe_path(workspace: str | Path, candidate: str | Path) -> Path | None:
    """Soft variant: returns ``None`` instead of raising on escape."""
    try:
        return resolve_within(workspace, candidate)
    except PathEscape:
        return None


__all__ = ["PathEscape", "resolve_within", "safe_path"]
