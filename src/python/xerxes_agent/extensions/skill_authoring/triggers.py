# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
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
"""Trigger heuristic for autonomous skill authoring.

Decides whether a finished :class:`SkillCandidate` is "skill-worthy".
Hermes uses ~5+ tool calls + success + novelty as its bar; Xerxes's
trigger exposes the thresholds via :class:`SkillAuthoringConfig`.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

from .tracker import SkillCandidate

if tp.TYPE_CHECKING:
    from ..skills import SkillRegistry


@dataclass
class SkillAuthoringConfig:
    """Knobs for the trigger heuristic.

    Attributes:
        min_tool_calls: Minimum events in the candidate (Hermes default: 5).
        require_success: Require zero non-success events when ``True``.
        max_retries_allowed: Reject candidates with more retries than this.
        min_unique_tools: Reject candidates that hammer one tool repeatedly.
        skip_if_skill_signature_exists: Avoid drafting if the registry
            already has a skill whose ``tags`` cover the same tools.
        enabled: Master switch.
    """

    min_tool_calls: int = 5
    require_success: bool = True
    max_retries_allowed: int = 2
    min_unique_tools: int = 2
    skip_if_skill_signature_exists: bool = True
    enabled: bool = True


class SkillAuthoringTrigger:
    """Pure-function trigger evaluator.

    Holds the optional skill registry reference for novelty checks.

    Example:
        >>> trig = SkillAuthoringTrigger(SkillAuthoringConfig())
        >>> if trig.should_author(candidate):
        ...     drafter.draft(candidate)
    """

    def __init__(
        self,
        config: SkillAuthoringConfig | None = None,
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        """Initialise with optional config and registry."""
        self.config = config or SkillAuthoringConfig()
        self.skill_registry = skill_registry

    def should_author(self, candidate: SkillCandidate) -> bool:
        """Return ``True`` if *candidate* meets the authoring threshold."""
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
        """Human-readable explanation of the trigger decision."""
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
        """Indices of failures that were never recovered by a later retry."""
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
        """Return ``True`` if the registry already covers this tool set."""
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
