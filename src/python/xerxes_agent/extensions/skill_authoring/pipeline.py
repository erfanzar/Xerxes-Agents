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
"""End-to-end Hermes-style skill authoring pipeline.

Wires together the per-iteration components into one orchestrator:

    on_turn_end ->
        ToolSequenceTracker.end_turn ->
        SkillAuthoringTrigger.should_author? ->
        SkillDrafter.draft ->
        SkillVerifier.generate (sidecar) ->
        AuditEmitter.emit_skill_authored ->
        return SkillSuggestion event

The pipeline is the natural integration point for the runtime — the
streaming loop calls :meth:`on_turn_end` at the end of every turn.
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
    """What the pipeline produced from one turn.

    Attributes:
        candidate: The captured tool sequence.
        authored: Whether a new skill was drafted.
        skill_path: Filesystem path of the new SKILL.md (when ``authored``).
        skill_name: Name of the drafted skill.
        version: Version of the drafted skill.
        recipe_path: Filesystem path of the verification recipe sidecar.
        reason: Human-readable explanation when not authored.
    """

    candidate: SkillCandidate
    authored: bool = False
    skill_path: Path | None = None
    skill_name: str = ""
    version: str = ""
    recipe_path: Path | None = None
    reason: str = ""


class SkillAuthoringPipeline:
    """End-to-end orchestrator for autonomous skill authoring.

    Holds references to the tracker, trigger, drafter, verifier, and
    telemetry. The runtime calls :meth:`begin_turn` / :meth:`record_call`
    during execution and :meth:`on_turn_end` to finalise.

    Example:
        >>> p = SkillAuthoringPipeline(skills_dir="./skills")
        >>> p.begin_turn(agent_id="coder", user_prompt="set up CI")
        >>> p.record_call("Read", {"path": "ci.yml"})
        >>> # ... 4 more calls ...
        >>> result = p.on_turn_end(final_response="CI configured")
        >>> if result.authored:
        ...     print(f"new skill: {result.skill_path}")
    """

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
        """Initialise the pipeline.

        Args:
            skills_dir: Directory where new SKILL.md files are written.
            config: Trigger heuristic config. Defaults to
                :class:`SkillAuthoringConfig` defaults.
            skill_registry: Optional registry for novelty checks; new
                skills are also registered here when authored.
            llm_client: Optional LLM for skill draft refinement.
            telemetry: Optional shared :class:`SkillTelemetry`.
            audit_emitter: Optional :class:`AuditEmitter` for events.
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
        """Start a tracked turn (delegates to the underlying tracker)."""
        self.tracker.begin_turn(agent_id=agent_id, turn_id=turn_id, user_prompt=user_prompt)

    def record_call(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        """Forward a tool-call recording to the tracker."""
        return self.tracker.record_call(*args, **kwargs)

    def on_turn_end(self, final_response: str = "") -> AuthoringResult:
        """Finalise the turn and run the authoring loop.

        Args:
            final_response: The agent's final answer for the turn.

        Returns:
            An :class:`AuthoringResult` describing what happened.
        """
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
        """Return the ``name:`` value from SKILL.md front-matter.

        Args:
            text: Raw contents of a SKILL.md file.

        Returns:
            The unquoted skill name, or ``None`` if no name line is present.
        """
        for line in text.splitlines():
            if line.startswith("name:"):
                return line.split(":", 1)[1].strip().strip('"').strip("'")
        return None

    @staticmethod
    def _extract_version(text: str) -> str | None:
        """Return the ``version:`` value from SKILL.md front-matter.

        Args:
            text: Raw contents of a SKILL.md file.

        Returns:
            The unquoted version string, or ``None`` if no version line is present.
        """
        for line in text.splitlines():
            if line.startswith("version:"):
                return line.split(":", 1)[1].strip().strip('"').strip("'")
        return None
