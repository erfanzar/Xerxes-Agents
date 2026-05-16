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
"""`/rollback diff [n]` helper — preview filesystem changes since a snapshot.

Uses the shadow git repo behind ``SnapshotManager`` to surface a textual
diff between the workspace's current state and a chosen snapshot, so the
user can decide before running ``mgr.rollback``."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

from .snapshots import SnapshotManager, SnapshotRecord


@dataclass
class SnapshotDiff:
    """Textual diff vs a snapshot.

    Attributes:
        snapshot: the snapshot that was diffed against.
        diff_text: ``git diff`` output (or empty if no changes).
        file_count: number of files with differences.
        added: lines added.
        removed: lines removed."""

    snapshot: SnapshotRecord
    diff_text: str
    file_count: int = 0
    added: int = 0
    removed: int = 0


def diff_against_snapshot(manager: SnapshotManager, ref: str) -> SnapshotDiff:
    """Return a ``SnapshotDiff`` between the current workspace and ``ref``."""
    snap = manager.get(ref)
    if snap is None:
        raise KeyError(f"snapshot not found: {ref}")
    out = _run_git(manager, ["diff", snap.commit_sha, "--", "."])
    diff_text = out
    file_count, added, removed = _summarize_diff(diff_text)
    return SnapshotDiff(snapshot=snap, diff_text=diff_text, file_count=file_count, added=added, removed=removed)


def _run_git(manager: SnapshotManager, args: list[str]) -> str:
    shadow = manager._shadow_path()
    env = os.environ.copy()
    env.update(
        {
            "GIT_DIR": str(shadow / ".git"),
            "GIT_WORK_TREE": str(manager.workspace),
        }
    )
    proc = subprocess.run(
        ["git", *args],
        cwd=str(manager.workspace),
        env=env,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def _summarize_diff(diff_text: str) -> tuple[int, int, int]:
    files = 0
    added = 0
    removed = 0
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            files += 1
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return files, added, removed


__all__ = ["SnapshotDiff", "diff_against_snapshot"]
