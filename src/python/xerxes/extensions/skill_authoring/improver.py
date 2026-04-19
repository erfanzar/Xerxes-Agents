# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.
"""Skill improver — bumps version & rewrites SKILL.md from a fresh candidate.

Triggered when telemetry shows a skill is failing too often. The improver
takes the original SKILL.md path and a recent (better) :class:`SkillCandidate`,
re-renders the body, bumps the patch version, and writes the result back
in-place. A backup of the previous version is left as ``SKILL.md.<old>.bak``
so improvements remain reviewable.
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
    """Outcome of a single improvement attempt.

    Attributes:
        improved: Whether a new version was written.
        old_version: Version before the rewrite.
        new_version: Version after the rewrite.
        backup_path: Path to the ``.bak`` of the prior SKILL.md.
        skill_path: Path to the rewritten SKILL.md.
        reason: Explanation when ``improved=False``.
    """

    improved: bool
    old_version: str = ""
    new_version: str = ""
    backup_path: Path | None = None
    skill_path: Path | None = None
    reason: str = ""


def _bump_patch(version: str) -> str:
    """Increment the patch component of a semver string.

    Falls back to ``0.1.1`` when the version is unparseable.
    """
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)", version.strip())
    if not m:
        return "0.1.1"
    major, minor, patch = m.group(1), m.group(2), int(m.group(3))
    return f"{major}.{minor}.{patch + 1}"


class SkillImprover:
    """Re-draft a SKILL.md from a higher-quality candidate.

    Use when telemetry flags a skill (e.g. via
    :meth:`SkillTelemetry.candidates_for_deprecation`) and the agent has
    since produced a successful run of the same procedure.

    Example:
        >>> imp = SkillImprover()
        >>> result = imp.improve(skill_path, fresh_candidate)
        >>> if result.improved:
        ...     print(f"{result.old_version} → {result.new_version}")
    """

    def improve(
        self,
        skill_path: str | Path,
        candidate: SkillCandidate,
        *,
        max_age_attempts: int = 5,
    ) -> ImprovementResult:
        """Re-render ``skill_path`` with content derived from *candidate*.

        Args:
            skill_path: Existing SKILL.md to rewrite.
            candidate: Fresh, successful tool sequence.
            max_age_attempts: Max number of retained ``.bak`` files
                before we start overwriting the oldest one.

        Returns:
            An :class:`ImprovementResult` describing the outcome.
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
        """Return the ``version:`` value from a SKILL.md front-matter block.

        Args:
            text: Raw contents of a SKILL.md file.

        Returns:
            The unquoted version string, or ``None`` if no version line is found.
        """
        for line in text.splitlines():
            if line.strip().startswith("version:"):
                return line.split(":", 1)[1].strip().strip('"').strip("'")
        return None

    @staticmethod
    def _extract_name(text: str) -> str | None:
        """Return the ``name:`` value from a SKILL.md front-matter block.

        Args:
            text: Raw contents of a SKILL.md file.

        Returns:
            The unquoted skill name, or ``None`` if no name line is found.
        """
        for line in text.splitlines():
            if line.strip().startswith("name:"):
                return line.split(":", 1)[1].strip().strip('"').strip("'")
        return None

    @staticmethod
    def _make_backup(path: Path, old_version: str, max_keep: int) -> Path:
        """Write ``SKILL.md.<old_version>.bak`` next to *path*; prune oldest."""
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
