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
"""Trigger logic that decides whether a tool sequence is worth authoring.

``SkillAuthoringTrigger`` applies configurable thresholds (minimum tool calls,
retry limits, uniqueness requirements) to a ``SkillCandidate``.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

from .tracker import SkillCandidate

if tp.TYPE_CHECKING:
    from ..skills import SkillRegistry


@dataclass
class SkillAuthoringConfig:
    """Thresholds for auto-authoring a skill from a turn.

    Attributes:
        min_tool_calls (int): IN: Minimum events required. OUT: Checked by
            ``should_author``.
        require_success (bool): IN: Whether failures disqualify. OUT: Checked
            by ``should_author``.
        max_retries_allowed (int): IN: Retry tolerance. OUT: Checked by
            ``should_author``.
        min_unique_tools (int): IN: Minimum distinct tools. OUT: Checked by
            ``should_author``.
        skip_if_skill_signature_exists (bool): IN: Duplicate detection. OUT:
            Checked by ``should_author``.
        enabled (bool): IN: Master switch. OUT: Checked by ``should_author``.
    """

    min_tool_calls: int = 5
    require_success: bool = True
    max_retries_allowed: int = 2
    min_unique_tools: int = 2
    skip_if_skill_signature_exists: bool = True
    enabled: bool = True


class SkillAuthoringTrigger:
    """Evaluates a ``SkillCandidate`` against authoring rules."""

    def __init__(
        self,
        config: SkillAuthoringConfig | None = None,
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        """Initialize with config and optional registry for duplicate checks.

        Args:
            config (SkillAuthoringConfig | None): IN: Thresholds. OUT: Defaults
                to a new ``SkillAuthoringConfig``.
            skill_registry (SkillRegistry | None): IN: Registry for duplicate
                detection. OUT: Used by ``_has_existing_skill``.
        """

        self.config = config or SkillAuthoringConfig()
        self.skill_registry = skill_registry

    def should_author(self, candidate: SkillCandidate) -> bool:
        """Return whether ``candidate`` meets all authoring thresholds.

        Args:
            candidate (SkillCandidate): IN: Observed turn data. OUT: Evaluated
                against every config threshold.

        Returns:
            bool: OUT: ``True`` if the candidate qualifies for skill creation.
        """

        cfg = self.config
        if not cfg.enabled:
            return False
        if len(candidate.events) < cfg.min_tool_calls:
            return False
        if cfg.require_success:
            if any(e.status != "success" for e in candidate.events if e.retry_of is None):
                pass
            if not candidate.successful_events:
                return False
            terminal_failures = self._terminal_failures(candidate)
            if terminal_failures:
                return False
        if candidate.retries > cfg.max_retries_allowed:
            return False
        if len(candidate.unique_tools) < cfg.min_unique_tools:
            return False
        if cfg.skip_if_skill_signature_exists and self._has_existing_skill(candidate):
            return False
        return True

    def reason(self, candidate: SkillCandidate) -> str:
        """Return a human-readable explanation of the authoring decision.

        Args:
            candidate (SkillCandidate): IN: Observed turn data. OUT: Evaluated
                to produce the message.

        Returns:
            str: OUT: Reason string; ``"skill-worthy"`` if all checks pass.
        """

        cfg = self.config
        if not cfg.enabled:
            return "skill authoring disabled"
        if len(candidate.events) < cfg.min_tool_calls:
            return f"only {len(candidate.events)} tool calls (< {cfg.min_tool_calls})"
        if cfg.require_success and self._terminal_failures(candidate):
            return "candidate has unrecovered failures"
        if candidate.retries > cfg.max_retries_allowed:
            return f"{candidate.retries} retries (> {cfg.max_retries_allowed})"
        if len(candidate.unique_tools) < cfg.min_unique_tools:
            return f"{len(candidate.unique_tools)} unique tools (< {cfg.min_unique_tools})"
        if cfg.skip_if_skill_signature_exists and self._has_existing_skill(candidate):
            return "an existing skill already covers this tool combination"
        return "skill-worthy"

    def _terminal_failures(self, candidate: SkillCandidate) -> list[int]:
        """Identify event indices that failed and were not later retried.

        Args:
            candidate (SkillCandidate): IN: Observed turn data. OUT: Scanned
                for failures and recoveries.

        Returns:
            list[int]: OUT: Indices of unrecovered failures.
        """

        recovered: set[int] = set()
        for ev in candidate.events:
            if ev.retry_of is not None and ev.status == "success":
                recovered.add(ev.retry_of)
        out = []
        for i, ev in enumerate(candidate.events):
            if ev.status == "success":
                continue
            if i in recovered:
                continue
            out.append(i)
        return out

    def _has_existing_skill(self, candidate: SkillCandidate) -> bool:
        """Check whether a skill already covers the candidate's tool set.

        Args:
            candidate (SkillCandidate): IN: Observed turn data. OUT: Compared
                against registry skills.

        Returns:
            bool: OUT: ``True`` if an existing skill's tags or required_tools
            superset the candidate's unique tools.
        """

        if self.skill_registry is None:
            return False
        try:
            skills = self.skill_registry.get_all()
        except Exception:
            return False
        candidate_tools = set(candidate.unique_tools)
        for skill in skills:
            tags = set(getattr(skill.metadata, "tags", []) or [])
            required = set(getattr(skill.metadata, "required_tools", []) or [])
            covered = tags | required
            if covered and candidate_tools.issubset(covered):
                return True
        return False
