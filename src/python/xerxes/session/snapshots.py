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
"""Filesystem snapshots via a shadow git repo.

Each project workspace gets a separate "shadow" git repo at
``$XERXES_HOME/snapshots/<workspace_hash>/`` that mirrors the real
working tree but does NOT touch the user's actual git history.
``snapshot()`` runs ``git add -A && git commit`` against the shadow
repo; ``rollback()`` does ``git checkout`` from a labeled commit back
into the real working tree."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .._compat_shims import xerxes_subdir_safe


@dataclass
class SnapshotRecord:
    """One filesystem snapshot.

    Attributes:
        id: stable record id (uuid).
        label: human label supplied by the user.
        commit_sha: shadow repo commit.
        created_at: ISO timestamp.
        workspace_dir: directory the snapshot applies to."""

    id: str
    label: str
    commit_sha: str
    created_at: str
    workspace_dir: str


class SnapshotManager:
    """Manage a shadow git repo per workspace directory."""

    def __init__(self, workspace_dir: str | Path, *, shadow_root: Path | None = None) -> None:
        self.workspace = Path(workspace_dir).resolve()
        self._shadow_root = shadow_root or xerxes_subdir_safe("snapshots")
        self._records_path = self._shadow_path() / "_records.txt"

    def _hash_workspace(self) -> str:
        return hashlib.sha1(str(self.workspace).encode()).hexdigest()[:12]

    def _shadow_path(self) -> Path:
        return self._shadow_root / self._hash_workspace()

    def _run_git(self, *args: str) -> subprocess.CompletedProcess:
        shadow = self._shadow_path()
        # Use --git-dir + --work-tree so the shadow git repo points at the
        # real working directory without leaving a ``.git`` there.
        env = os.environ.copy()
        env.update(
            {
                "GIT_DIR": str(shadow / ".git"),
                "GIT_WORK_TREE": str(self.workspace),
                # Make the commit author deterministic enough for tests.
                "GIT_AUTHOR_NAME": "xerxes-snapshot",
                "GIT_AUTHOR_EMAIL": "snapshots@xerxes",
                "GIT_COMMITTER_NAME": "xerxes-snapshot",
                "GIT_COMMITTER_EMAIL": "snapshots@xerxes",
            }
        )
        return subprocess.run(
            ["git", *args],
            cwd=str(self.workspace),
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

    def _ensure_repo(self) -> None:
        shadow = self._shadow_path()
        if not (shadow / ".git").exists():
            shadow.mkdir(parents=True, exist_ok=True)
            self._run_git("init", "--quiet", "--initial-branch", "main", str(shadow / ".git"))
            # Configure to ignore the shadow root itself if present in workspace.
            (shadow / ".git" / "info").mkdir(parents=True, exist_ok=True)
            (shadow / ".git" / "info" / "exclude").write_text(".xerxes/snapshots/**\n")

    # ---------------------------- public surface

    def snapshot(self, label: str = "") -> SnapshotRecord:
        """Commit the current workspace state to the shadow repo."""
        self._ensure_repo()
        self._run_git("add", "-A")
        msg = label or f"snapshot-{datetime.now(UTC).isoformat()}"
        commit = self._run_git("commit", "--allow-empty", "-m", msg)
        if commit.returncode != 0:
            raise RuntimeError(f"snapshot commit failed: {commit.stderr.strip()}")
        sha = self._run_git("rev-parse", "HEAD").stdout.strip()
        record = SnapshotRecord(
            id=uuid.uuid4().hex[:12],
            label=label,
            commit_sha=sha,
            created_at=datetime.now(UTC).isoformat(),
            workspace_dir=str(self.workspace),
        )
        self._append_record(record)
        return record

    def list(self) -> list[SnapshotRecord]:
        path = self._records_path
        if not path.exists():
            return []
        records: list[SnapshotRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 5:
                continue
            records.append(
                SnapshotRecord(
                    id=parts[0], label=parts[1], commit_sha=parts[2], created_at=parts[3], workspace_dir=parts[4]
                )
            )
        return records

    def get(self, ref: str) -> SnapshotRecord | None:
        for r in self.list():
            if r.id == ref or r.label == ref or r.commit_sha.startswith(ref):
                return r
        return None

    def rollback(self, ref: str) -> SnapshotRecord:
        """Restore the workspace files to the snapshot ``ref``.

        Uses ``git checkout <sha> -- .`` so it touches files but not the
        shadow repo's branch state. Files added since the snapshot are
        NOT removed (only files tracked by the snapshot are restored);
        callers wanting a hard reset should manually rm extras first."""
        record = self.get(ref)
        if record is None:
            raise KeyError(f"snapshot not found: {ref}")
        out = self._run_git("checkout", record.commit_sha, "--", ".")
        if out.returncode != 0:
            raise RuntimeError(f"rollback failed: {out.stderr.strip()}")
        return record

    def prune(self, *, keep: int = 100) -> int:
        """Drop record entries beyond the most recent ``keep``."""
        records = self.list()
        if len(records) <= keep:
            return 0
        to_keep = records[-keep:]
        path = self._records_path
        path.write_text(
            "\n".join("\t".join((r.id, r.label, r.commit_sha, r.created_at, r.workspace_dir)) for r in to_keep),
            encoding="utf-8",
        )
        return len(records) - keep

    def _append_record(self, record: SnapshotRecord) -> None:
        path = self._records_path
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        new_line = "\t".join((record.id, record.label, record.commit_sha, record.created_at, record.workspace_dir))
        path.write_text(
            existing + ("\n" if existing and not existing.endswith("\n") else "") + new_line + "\n", encoding="utf-8"
        )

    def reset(self) -> None:
        """Remove the shadow repo (irreversible). Used by tests."""
        shadow = self._shadow_path()
        if shadow.exists():
            shutil.rmtree(shadow)


__all__ = ["SnapshotManager", "SnapshotRecord"]
