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
"""End-to-end skill authoring pipeline.

``SkillAuthoringPipeline`` wires together tracking, triggering, drafting,
verification, and telemetry so that a completed agent turn can automatically
yield a new ``SKILL.md``.
"""

from __future__ import annotations

import json
import logging
import typing as tp
from dataclasses import dataclass
from pathlib import Path

from .drafter import SkillDrafter
from .telemetry import SkillTelemetry
from .tracker import SkillCandidate, ToolSequenceTracker
from .triggers import SkillAuthoringConfig, SkillAuthoringTrigger
from .verifier import SkillVerifier

if tp.TYPE_CHECKING:
    from ...audit.emitter import AuditEmitter
    from ..skills import SkillRegistry
logger = logging.getLogger(__name__)


@dataclass
class AuthoringResult:
    """Outcome of a single turn's auto-authoring attempt.

    Attributes:
        candidate: The observed tool sequence considered for authoring.
        authored: True when a new skill was successfully written.
        skill_path: Path of the written ``SKILL.md`` on success.
        skill_name: Skill name extracted from the rendered frontmatter.
        version: Version extracted from the rendered frontmatter.
        recipe_path: Path of the verification recipe JSON, when produced.
        reason: Empty on success; explanatory message on failure or skip.
    """

    candidate: SkillCandidate
    authored: bool = False
    skill_path: Path | None = None
    skill_name: str = ""
    version: str = ""
    recipe_path: Path | None = None
    reason: str = ""


class SkillAuthoringPipeline:
    """Orchestrate automatic skill creation around an agent turn."""

    def __init__(
        self,
        skills_dir: str | Path,
        *,
        config: SkillAuthoringConfig | None = None,
        skill_registry: SkillRegistry | None = None,
        llm_client: tp.Any | None = None,
        telemetry: SkillTelemetry | None = None,
        audit_emitter: AuditEmitter | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            skills_dir: Destination directory for drafted skills.
            config: Trigger thresholds; defaults to ``SkillAuthoringConfig()``.
            skill_registry: Registry consulted to skip duplicate skills.
            llm_client: Optional LLM client used to refine drafts.
            telemetry: Telemetry backend; defaults to a new ``SkillTelemetry``.
            audit_emitter: Emitter used to log ``SkillAuthored`` events.
        """
        self.tracker = ToolSequenceTracker()
        self.config = config or SkillAuthoringConfig()
        self.trigger = SkillAuthoringTrigger(self.config, skill_registry=skill_registry)
        self.drafter = SkillDrafter(skills_dir, llm_client=llm_client)
        self.verifier = SkillVerifier()
        self.telemetry = telemetry or SkillTelemetry()
        self.audit_emitter = audit_emitter
        self.skill_registry = skill_registry

    def begin_turn(
        self,
        agent_id: str | None = None,
        turn_id: str | None = None,
        user_prompt: str = "",
    ) -> None:
        """Reset the internal tracker for a new agent turn."""

        self.tracker.begin_turn(agent_id=agent_id, turn_id=turn_id, user_prompt=user_prompt)

    def record_call(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        """Forward a tool call event to the internal tracker."""

        return self.tracker.record_call(*args, **kwargs)

    def on_turn_end(self, final_response: str = "") -> AuthoringResult:
        """End the turn and author a skill when triggers and verification succeed."""

        candidate = self.tracker.end_turn(final_response=final_response)
        if not self.trigger.should_author(candidate):
            return AuthoringResult(
                candidate=candidate,
                authored=False,
                reason=self.trigger.reason(candidate),
            )
        try:
            text, path = self.drafter.draft(candidate)
        except Exception:
            logger.warning("SkillDrafter.draft failed", exc_info=True)
            return AuthoringResult(candidate=candidate, authored=False, reason="drafter raised")
        skill_name = self._extract_name(text) or candidate.signature() or "unnamed"
        version = self._extract_version(text) or "0.1.0"
        recipe_path: Path | None = None
        if path is not None:
            try:
                steps = self.verifier.generate(candidate)
                recipe_path = path.with_suffix(".verify.json")
                recipe_path.write_text(
                    json.dumps([s.__dict__ for s in steps], indent=2, default=str),
                    encoding="utf-8",
                )
            except Exception:
                logger.debug("Verification recipe write failed", exc_info=True)
                recipe_path = None
        if self.audit_emitter is not None:
            try:
                self.audit_emitter.emit_skill_authored(
                    skill_name=skill_name,
                    version=version,
                    source_path=str(path) if path else "",
                    tool_count=len(candidate.events),
                    unique_tools=candidate.unique_tools,
                    confirmed_by_user=False,
                    agent_id=candidate.agent_id,
                    turn_id=candidate.turn_id,
                )
            except Exception:
                logger.debug("emit_skill_authored failed", exc_info=True)
        return AuthoringResult(
            candidate=candidate,
            authored=True,
            skill_path=path,
            skill_name=skill_name,
            version=version,
            recipe_path=recipe_path,
        )

    @staticmethod
    def _extract_name(text: str) -> str | None:
        """Return the frontmatter ``name`` value, or ``None``."""

        for line in text.splitlines():
            if line.startswith("name:"):
                return line.split(":", 1)[1].strip().strip('"').strip("'")
        return None

    @staticmethod
    def _extract_version(text: str) -> str | None:
        """Return the frontmatter ``version`` value, or ``None``."""

        for line in text.splitlines():
            if line.startswith("version:"):
                return line.split(":", 1)[1].strip().strip('"').strip("'")
        return None
