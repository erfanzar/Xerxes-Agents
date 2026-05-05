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
"""Incremental skill improvement from new tool sequences.

``SkillImprover`` bumps the patch version of an existing ``SKILL.md`` and
rewrites it using a newer ``SkillCandidate``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .drafter import render_skill_template
from .tracker import SkillCandidate

logger = logging.getLogger(__name__)


@dataclass
class ImprovementResult:
    """Outcome of a skill improvement attempt.

    Attributes:
        improved (bool): IN: Success flag. OUT: ``True`` if the file was
            updated.
        old_version (str): IN: Previous version. OUT: Extracted from existing
            file.
        new_version (str): IN: Bumped version. OUT: Computed by
            ``_bump_patch``.
        backup_path (Path | None): IN: Path to backup. OUT: Set if a backup
            was created.
        skill_path (Path | None): IN: Updated file path. OUT: Set on success.
        reason (str): IN: Empty on success. OUT: Error description on failure.
    """

    improved: bool
    old_version: str = ""
    new_version: str = ""
    backup_path: Path | None = None
    skill_path: Path | None = None
    reason: str = ""


def _bump_patch(version: str) -> str:
    """Increment the patch component of a semver string.

    Args:
        version (str): IN: Semver-like string. OUT: Parsed and incremented.

    Returns:
        str: OUT: New version string (defaults to ``"0.1.1"`` if parsing
        fails).
    """

    m = re.match(r"^(\d+)\.(\d+)\.(\d+)", version.strip())
    if not m:
        return "0.1.1"
    major, minor, patch = m.group(1), m.group(2), int(m.group(3))
    return f"{major}.{minor}.{patch + 1}"


class SkillImprover:
    """Rewrite an existing skill with an improved tool sequence."""

    def improve(
        self,
        skill_path: str | Path,
        candidate: SkillCandidate,
        *,
        max_age_attempts: int = 5,
    ) -> ImprovementResult:
        """Update an existing ``SKILL.md`` from a new candidate.

        Args:
            skill_path (str | Path): IN: Path to the existing file. OUT: Read
                and overwritten.
            candidate (SkillCandidate): IN: New observed sequence. OUT: Passed
                to ``render_skill_template``.
            max_age_attempts (int): IN: Max backups to retain. OUT: Passed to
                ``_make_backup``.

        Returns:
            ImprovementResult: OUT: Detailed result of the operation.
        """

        path = Path(skill_path).expanduser()
        if not path.exists():
            return ImprovementResult(improved=False, reason=f"missing skill at {path}")
        try:
            old_text = path.read_text(encoding="utf-8")
        except Exception:
            return ImprovementResult(improved=False, reason="failed to read existing SKILL.md")
        old_version = self._extract_version(old_text) or "0.1.0"
        new_version = _bump_patch(old_version)
        old_name = self._extract_name(old_text) or path.parent.name
        try:
            new_text = render_skill_template(
                candidate,
                name=old_name,
                version=new_version,
            )
        except Exception:
            return ImprovementResult(improved=False, reason="render_skill_template raised")
        try:
            backup_path = self._make_backup(path, old_version, max_age_attempts)
            path.write_text(new_text, encoding="utf-8")
        except Exception:
            return ImprovementResult(
                improved=False,
                old_version=old_version,
                new_version=new_version,
                reason="failed to write SKILL.md",
            )
        return ImprovementResult(
            improved=True,
            old_version=old_version,
            new_version=new_version,
            backup_path=backup_path,
            skill_path=path,
        )

    @staticmethod
    def _extract_version(text: str) -> str | None:
        """Parse the ``version`` field from YAML frontmatter.

        Args:
            text (str): IN: SKILL.md content. OUT: Searched line-by-line.

        Returns:
            str | None: OUT: Version string or ``None``.
        """

        for line in text.splitlines():
            if line.strip().startswith("version:"):
                return line.split(":", 1)[1].strip().strip('"').strip("'")
        return None

    @staticmethod
    def _extract_name(text: str) -> str | None:
        """Parse the ``name`` field from YAML frontmatter.

        Args:
            text (str): IN: SKILL.md content. OUT: Searched line-by-line.

        Returns:
            str | None: OUT: Name string or ``None``.
        """

        for line in text.splitlines():
            if line.strip().startswith("name:"):
                return line.split(":", 1)[1].strip().strip('"').strip("'")
        return None

    @staticmethod
    def _make_backup(path: Path, old_version: str, max_keep: int) -> Path:
        """Create a versioned backup and prune old ones.

        Args:
            path (Path): IN: File to back up. OUT: Copied.
            old_version (str): IN: Version for the backup filename. OUT: Used
                in suffix.
            max_keep (int): IN: Maximum backups to retain. OUT: Older backups
                are deleted.

        Returns:
            Path: OUT: Path to the newly created backup file.
        """

        backup = path.with_name(f"{path.name}.{old_version}.bak")
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        backups = sorted(path.parent.glob(f"{path.name}.*.bak"))
        if len(backups) > max_keep:
            for old_b in backups[:-max_keep]:
                try:
                    old_b.unlink()
                except Exception:
                    pass
        return backup
