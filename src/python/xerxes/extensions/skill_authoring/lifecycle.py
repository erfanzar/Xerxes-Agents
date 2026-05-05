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

``SkillVariantPicker`` supports canary rollouts. ``SkillLifecycleManager"
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
        base_name (str): IN: Original skill name. OUT: Lookup key.
        variant_name (str): IN: Alternative skill name. OUT: Returned by
            ``pick`` for the rollout bucket.
        rollout (float): IN: Proportion [0.0, 1.0] of users receiving the
            variant. OUT: Clamped in ``__post_init__``.
    """

    base_name: str
    variant_name: str
    rollout: float = 0.5

    def __post_init__(self) -> None:
        """Clamp ``rollout`` to the [0.0, 1.0] range."""

        self.rollout = max(0.0, min(1.0, float(self.rollout)))


class SkillVariantPicker:
    """Deterministically assigns users to skill variants via hash bucketing."""

    def __init__(self) -> None:
        """Initialize an empty variant index."""

        self._variants: dict[str, SkillVariant] = {}

    def add(self, variant: SkillVariant) -> None:
        """Register a variant.

        Args:
            variant (SkillVariant): IN: Variant definition. OUT: Stored by
                ``base_name``.

        Returns:
            None: OUT: Overwrites any existing variant for the same base.
        """

        self._variants[variant.base_name] = variant

    def remove(self, base_name: str) -> None:
        """Remove a variant entry.

        Args:
            base_name (str): IN: Skill base name. OUT: Key removed from the
                index.

        Returns:
            None: OUT: Variant is no longer tracked.
        """

        self._variants.pop(base_name, None)

    def get(self, base_name: str) -> SkillVariant | None:
        """Retrieve a variant definition.

        Args:
            base_name (str): IN: Skill base name. OUT: Looked up in the
                index.

        Returns:
            SkillVariant | None: OUT: Variant record or ``None``.
        """

        return self._variants.get(base_name)

    def pick(self, base_name: str, user_id: str = "") -> str:
        """Select either the base skill or the variant based on rollout.

        Bucketing is deterministic given ``user_id`` and ``base_name``.

        Args:
            base_name (str): IN: Skill name to resolve. OUT: Looked up in the
                variant index.
            user_id (str): IN: User identifier for consistent bucketing. OUT:
                Hashed together with ``base_name``.

        Returns:
            str: OUT: ``variant_name`` or ``base_name`` depending on bucket.
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
        """Return a snapshot of all registered variants.

        Returns:
            dict[str, SkillVariant]: OUT: Copy of the internal mapping.
        """

        return dict(self._variants)


@dataclass
class DeprecationDecision:
    """Outcome of a single skill deprecation evaluation.

    Attributes:
        skill_name (str): IN: Skill identifier. OUT: Stored.
        action (str): IN: Decision label (proposed, deprecated, kept, missing).
            OUT: Stored.
        reason (str): IN: Human-readable rationale. OUT: Stored.
        deprecated_path (Path | None): IN: New path after rename. OUT: Set
            when deprecation is applied.
    """

    skill_name: str
    action: str
    reason: str = ""
    deprecated_path: Path | None = None


class SkillLifecycleManager:
    """Evaluate telemetry and deprecate under-performing skills.

    Args:
        telemetry (SkillTelemetry): IN: Telemetry backend. OUT: Queried for
            success rates and invocation counts.
        registry (SkillRegistry | None): IN: Skill registry. OUT: Used to
            remove deprecated skills.
        skills_dir (str | Path | None): IN: Directory containing skills.
            OUT: Used as fallback when registry lookup fails.
        min_invocations (int): IN: Minimum invocations before a skill is
            eligible for deprecation. OUT: Passed to telemetry queries.
        max_success_rate (float): IN: Success rate threshold. OUT: Skills
            at or below this rate are flagged.
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
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            telemetry (SkillTelemetry): IN: telemetry. OUT: Consumed during execution.
            registry (SkillRegistry | None, optional): IN: registry. Defaults to None. OUT: Consumed during execution.
            skills_dir (str | Path | None, optional): IN: skills dir. Defaults to None. OUT: Consumed during execution.
            min_invocations (int, optional): IN: min invocations. Defaults to 10. OUT: Consumed during execution.
            max_success_rate (float, optional): IN: max success rate. Defaults to 0.4. OUT: Consumed during execution."""
        self.telemetry = telemetry
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            telemetry (SkillTelemetry): IN: telemetry. OUT: Consumed during execution.
            registry (SkillRegistry | None, optional): IN: registry. Defaults to None. OUT: Consumed during execution.
            skills_dir (str | Path | None, optional): IN: skills dir. Defaults to None. OUT: Consumed during execution.
            min_invocations (int, optional): IN: min invocations. Defaults to 10. OUT: Consumed during execution.
            max_success_rate (float, optional): IN: max success rate. Defaults to 0.4. OUT: Consumed during execution."""
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            telemetry (SkillTelemetry): IN: telemetry. OUT: Consumed during execution.
            registry (SkillRegistry | None, optional): IN: registry. Defaults to None. OUT: Consumed during execution.
            skills_dir (str | Path | None, optional): IN: skills dir. Defaults to None. OUT: Consumed during execution.
            min_invocations (int, optional): IN: min invocations. Defaults to 10. OUT: Consumed during execution.
            max_success_rate (float, optional): IN: max success rate. Defaults to 0.4. OUT: Consumed during execution."""
        self.registry = registry
        self.skills_dir = Path(skills_dir).expanduser() if skills_dir else None
        self.min_invocations = int(min_invocations)
        self.max_success_rate = float(max_success_rate)

    def evaluate(self) -> list[DeprecationDecision]:
        """Identify skills that qualify for deprecation.

        Returns:
            list[DeprecationDecision]: OUT: Proposed decisions with reasons.
        """

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
        """Rename qualifying skills to ``.deprecated.md`` and unregister them.

        Returns:
            list[DeprecationDecision]: OUT: Applied decisions reflecting the
            actual filesystem outcome.
        """

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
        """Find the ``SKILL.md`` path for a skill by name.

        Args:
            skill_name (str): IN: Skill identifier. OUT: Looked up in the
                registry, then on disk.

        Returns:
            Path | None: OUT: File path or ``None`` if not found.
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
