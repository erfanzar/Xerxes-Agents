# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.

"""Skill lifecycle: A/B variants + auto-deprecation.

Hermes runs continuous experiments on its authored skills:

- Two variants of the same skill (``base``/``variant``) can coexist;
  :class:`SkillVariantPicker` deterministically routes a given user
  to one of them.
- :class:`SkillLifecycleManager` periodically inspects telemetry and
  marks under-performing skills as deprecated (renames the SKILL.md
  file to ``SKILL.deprecated.md`` so the registry stops loading it).
"""

from __future__ import annotations

import hashlib
import logging
import typing as tp
from dataclasses import dataclass
from pathlib import Path

from .telemetry import SkillTelemetry

if tp.TYPE_CHECKING:
    from ..skills import SkillRegistry
logger = logging.getLogger(__name__)


@dataclass
class SkillVariant:
    """Definition of an A/B variant.

    Attributes:
        base_name: The canonical skill name being varied.
        variant_name: Identifier of the alternate skill.
        rollout: Fraction of users in ``[0, 1]`` routed to the variant.
    """

    base_name: str
    variant_name: str
    rollout: float = 0.5

    def __post_init__(self) -> None:
        """Clamp the rollout fraction to the ``[0.0, 1.0]`` interval."""
        self.rollout = max(0.0, min(1.0, float(self.rollout)))


class SkillVariantPicker:
    """Deterministic per-user variant routing.

    Hashes ``(user_id, base_name)`` and compares to the rollout
    threshold so the same user always sees the same variant.

    Example:
        >>> p = SkillVariantPicker()
        >>> p.add(SkillVariant("ci-setup", "ci-setup-v2", rollout=0.5))
        >>> p.pick("ci-setup", user_id="alice")
        'ci-setup-v2'
    """

    def __init__(self) -> None:
        """Initialise an empty variant registry."""
        self._variants: dict[str, SkillVariant] = {}

    def add(self, variant: SkillVariant) -> None:
        """Register *variant*, replacing any existing entry for its base name.

        Args:
            variant: The :class:`SkillVariant` to install.
        """
        self._variants[variant.base_name] = variant

    def remove(self, base_name: str) -> None:
        """Drop any variant registered under *base_name*.

        Args:
            base_name: Canonical skill name whose variant should be cleared.
        """
        self._variants.pop(base_name, None)

    def get(self, base_name: str) -> SkillVariant | None:
        """Return the variant registered for *base_name*, or ``None``.

        Args:
            base_name: Canonical skill name to look up.

        Returns:
            The matching :class:`SkillVariant`, or ``None`` if absent.
        """
        return self._variants.get(base_name)

    def pick(self, base_name: str, user_id: str = "") -> str:
        """Return either ``base_name`` or the configured ``variant_name``."""
        variant = self._variants.get(base_name)
        if variant is None or variant.rollout <= 0.0:
            return base_name
        if variant.rollout >= 1.0:
            return variant.variant_name
        h = hashlib.md5(f"{user_id}::{base_name}".encode()).digest()
        bucket = int.from_bytes(h[:4], "big") / 0xFFFFFFFF
        return variant.variant_name if bucket < variant.rollout else base_name

    def all(self) -> dict[str, SkillVariant]:
        """Return a snapshot of the ``base_name -> SkillVariant`` mapping."""
        return dict(self._variants)


@dataclass
class DeprecationDecision:
    """Outcome of one auto-deprecation pass.

    Attributes:
        skill_name: Skill that was evaluated.
        action: ``"deprecated"`` / ``"kept"`` / ``"missing"``.
        reason: Why the action was taken.
        deprecated_path: New filename when ``action == "deprecated"``.
    """

    skill_name: str
    action: str
    reason: str = ""
    deprecated_path: Path | None = None


class SkillLifecycleManager:
    """Periodic skill deprecation based on telemetry.

    Inspect :class:`SkillTelemetry` for skills below
    :attr:`telemetry_threshold_success_rate` over at least
    :attr:`min_invocations` invocations and, if a corresponding SKILL.md
    file is found, rename it to ``SKILL.deprecated.md``. The registry
    discovery routine is taught to skip ``*.deprecated.md`` so the
    skill is no longer loaded into prompts.
    """

    DEPRECATED_SUFFIX = ".deprecated.md"

    def __init__(
        self,
        telemetry: SkillTelemetry,
        registry: SkillRegistry | None = None,
        skills_dir: str | Path | None = None,
        *,
        min_invocations: int = 10,
        max_success_rate: float = 0.4,
    ) -> None:
        """Configure the lifecycle manager.

        Args:
            telemetry: Source of skill invocation statistics.
            registry: Optional :class:`SkillRegistry` to drop deprecated
                skills from in-memory.
            skills_dir: Root directory searched for SKILL.md files when
                the registry cannot locate one.
            min_invocations: Minimum invocation count before a skill is
                eligible for auto-deprecation.
            max_success_rate: Deprecate skills whose success rate is at
                or below this threshold.
        """
        self.telemetry = telemetry
        self.registry = registry
        self.skills_dir = Path(skills_dir).expanduser() if skills_dir else None
        self.min_invocations = int(min_invocations)
        self.max_success_rate = float(max_success_rate)

    def evaluate(self) -> list[DeprecationDecision]:
        """Inspect telemetry and produce decisions (no side effects)."""
        decisions: list[DeprecationDecision] = []
        candidates = self.telemetry.candidates_for_deprecation(
            min_invocations=self.min_invocations,
            max_success_rate=self.max_success_rate,
        )
        for name in candidates:
            stats = self.telemetry.stats(name)
            reason = (
                f"success_rate={stats.success_rate:.0%} after {stats.invocations} invocations"
                if stats
                else "stats unavailable"
            )
            decisions.append(DeprecationDecision(skill_name=name, action="proposed", reason=reason))
        return decisions

    def apply(self) -> list[DeprecationDecision]:
        """Evaluate AND deprecate matching skills (renames the SKILL.md)."""
        decisions = self.evaluate()
        applied: list[DeprecationDecision] = []
        for d in decisions:
            path = self._locate_skill(d.skill_name)
            if path is None:
                applied.append(
                    DeprecationDecision(skill_name=d.skill_name, action="missing", reason="SKILL.md not found")
                )
                continue
            new_path = path.with_name(path.name.replace(".md", self.DEPRECATED_SUFFIX))
            try:
                path.rename(new_path)
            except OSError as exc:
                logger.warning("Failed to deprecate %s: %s", path, exc)
                applied.append(
                    DeprecationDecision(skill_name=d.skill_name, action="kept", reason=f"rename failed: {exc}")
                )
                continue
            if self.registry is not None:
                try:
                    self.registry._skills.pop(d.skill_name, None)
                except Exception:
                    pass
            applied.append(
                DeprecationDecision(
                    skill_name=d.skill_name,
                    action="deprecated",
                    reason=d.reason,
                    deprecated_path=new_path,
                )
            )
        return applied

    def _locate_skill(self, skill_name: str) -> Path | None:
        """Resolve the SKILL.md path for *skill_name*.

        Consults the registry's ``source_path`` first, then falls back to
        a filename match under :attr:`skills_dir`, and finally scans
        nested SKILL.md files for a matching ``name:`` field.

        Args:
            skill_name: Canonical name of the skill to locate.

        Returns:
            Path to the SKILL.md file, or ``None`` if no match is found.
        """
        if self.registry is not None:
            try:
                skill = self.registry.get(skill_name)
                if skill and getattr(skill, "source_path", None):
                    return Path(skill.source_path)
            except Exception:
                pass
        if self.skills_dir is not None and self.skills_dir.is_dir():
            candidate = self.skills_dir / skill_name / "SKILL.md"
            if candidate.exists():
                return candidate
            for p in self.skills_dir.rglob("SKILL.md"):
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    if f"name: {skill_name}" in text:
                        return p
                except Exception:
                    continue
        return None


__all__ = [
    "DeprecationDecision",
    "SkillLifecycleManager",
    "SkillVariant",
    "SkillVariantPicker",
]
