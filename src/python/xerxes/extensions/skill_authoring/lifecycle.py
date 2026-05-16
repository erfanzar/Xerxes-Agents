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
"""Skill lifecycle management: variants, A/B rollout, and deprecation.

``SkillVariantPicker`` supports canary rollouts. ``SkillLifecycleManager``
evaluates telemetry to propose and apply deprecation of under-performing
skills.
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
    """A/B variant definition for a skill.

    Attributes:
        base_name: Original skill name (lookup key).
        variant_name: Alternative skill name returned for the rollout bucket.
        rollout: Proportion in [0.0, 1.0] of users routed to the variant.
    """

    base_name: str
    variant_name: str
    rollout: float = 0.5

    def __post_init__(self) -> None:
        """Clamp ``rollout`` to ``[0.0, 1.0]``."""

        self.rollout = max(0.0, min(1.0, float(self.rollout)))


class SkillVariantPicker:
    """Deterministically routes users between base skills and variants."""

    def __init__(self) -> None:
        """Initialize an empty variant index."""

        self._variants: dict[str, SkillVariant] = {}

    def add(self, variant: SkillVariant) -> None:
        """Register ``variant``, overwriting any prior entry under the same base name."""

        self._variants[variant.base_name] = variant

    def remove(self, base_name: str) -> None:
        """Drop the variant entry for ``base_name``, if any."""

        self._variants.pop(base_name, None)

    def get(self, base_name: str) -> SkillVariant | None:
        """Return the variant registered for ``base_name``, or ``None``."""

        return self._variants.get(base_name)

    def pick(self, base_name: str, user_id: str = "") -> str:
        """Return the name to use for ``base_name`` given ``user_id``.

        Bucketing is deterministic in ``user_id`` and ``base_name``, so the same
        user always lands in the same variant.
        """

        variant = self._variants.get(base_name)
        if variant is None or variant.rollout <= 0.0:
            return base_name
        if variant.rollout >= 1.0:
            return variant.variant_name
        h = hashlib.md5(f"{user_id}::{base_name}".encode()).digest()
        bucket = int.from_bytes(h[:4], "big") / 0xFFFFFFFF
        return variant.variant_name if bucket < variant.rollout else base_name

    def all(self) -> dict[str, SkillVariant]:
        """Return a copy of the variant index."""

        return dict(self._variants)


@dataclass
class DeprecationDecision:
    """Outcome of a single skill deprecation evaluation.

    Attributes:
        skill_name: Skill identifier.
        action: Decision label such as ``proposed``, ``deprecated``, ``kept``,
            or ``missing``.
        reason: Human-readable rationale.
        deprecated_path: Renamed file path once the decision is applied.
    """

    skill_name: str
    action: str
    reason: str = ""
    deprecated_path: Path | None = None


class SkillLifecycleManager:
    """Evaluate telemetry and deprecate under-performing skills."""

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
        """Initialize the lifecycle manager.

        Args:
            telemetry: Telemetry backend providing success rates and counts.
            registry: Optional skill registry; deprecated skills are pruned here.
            skills_dir: Directory of skill files used when registry lookup fails.
            min_invocations: Minimum invocations before deprecation eligibility.
            max_success_rate: Success-rate ceiling for flagging skills.
        """
        self.telemetry = telemetry
        self.registry = registry
        self.skills_dir = Path(skills_dir).expanduser() if skills_dir else None
        self.min_invocations = int(min_invocations)
        self.max_success_rate = float(max_success_rate)

    def evaluate(self) -> list[DeprecationDecision]:
        """Return proposed deprecation decisions for under-performing skills."""

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
        """Rename qualifying skills to ``.deprecated.md`` and remove them from the registry."""

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
        """Return the ``SKILL.md`` path for ``skill_name``, checking registry then disk."""

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
