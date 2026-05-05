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
    """Outcome of a single turn's authoring attempt.

    Attributes:
        candidate (SkillCandidate): IN: Observed tool sequence. OUT: Stored.
        authored (bool): IN: Success flag. OUT: ``True`` if a skill was
            written.
        skill_path (Path | None): IN: Written file path. OUT: Set on success.
        skill_name (str): IN: Extracted skill name. OUT: Set on success.
        version (str): IN: Extracted version. OUT: Set on success.
        recipe_path (Path | None): IN: Verification recipe path. OUT: Set on
            success.
        reason (str): IN: Empty on success. OUT: Failure description.
    """

    candidate: SkillCandidate
    authored: bool = False
    skill_path: Path | None = None
    skill_name: str = ""
    version: str = ""
    recipe_path: Path | None = None
    reason: str = ""


class SkillAuthoringPipeline:
    """Orchestrates automatic skill creation after agent turns.

    Args:
        skills_dir (str | Path): IN: Destination for drafted skills. OUT:
            Passed to ``SkillDrafter``.
        config (SkillAuthoringConfig | None): IN: Trigger thresholds. OUT:
            Defaults to a new ``SkillAuthoringConfig``.
        skill_registry (SkillRegistry | None): IN: Registry for duplicate
            detection. OUT: Passed to ``SkillAuthoringTrigger``.
        llm_client (tp.Any | None): IN: Optional LLM for refinement. OUT:
            Passed to ``SkillDrafter``.
        telemetry (SkillTelemetry | None): IN: Telemetry backend. OUT:
            Defaults to a new ``SkillTelemetry``.
        audit_emitter (AuditEmitter | None): IN: Optional audit emitter. OUT:
            Used to log authored events.
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
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            skills_dir (str | Path): IN: skills dir. OUT: Consumed during execution.
            config (SkillAuthoringConfig | None, optional): IN: config. Defaults to None. OUT: Consumed during execution.
            skill_registry (SkillRegistry | None, optional): IN: skill registry. Defaults to None. OUT: Consumed during execution.
            llm_client (tp.Any | None, optional): IN: llm client. Defaults to None. OUT: Consumed during execution.
            telemetry (SkillTelemetry | None, optional): IN: telemetry. Defaults to None. OUT: Consumed during execution.
            audit_emitter (AuditEmitter | None, optional): IN: audit emitter. Defaults to None. OUT: Consumed during execution."""
        self.tracker = ToolSequenceTracker()
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            skills_dir (str | Path): IN: skills dir. OUT: Consumed during execution.
            config (SkillAuthoringConfig | None, optional): IN: config. Defaults to None. OUT: Consumed during execution.
            skill_registry (SkillRegistry | None, optional): IN: skill registry. Defaults to None. OUT: Consumed during execution.
            llm_client (tp.Any | None, optional): IN: llm client. Defaults to None. OUT: Consumed during execution.
            telemetry (SkillTelemetry | None, optional): IN: telemetry. Defaults to None. OUT: Consumed during execution.
            audit_emitter (AuditEmitter | None, optional): IN: audit emitter. Defaults to None. OUT: Consumed during execution."""
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            skills_dir (str | Path): IN: skills dir. OUT: Consumed during execution.
            config (SkillAuthoringConfig | None, optional): IN: config. Defaults to None. OUT: Consumed during execution.
            skill_registry (SkillRegistry | None, optional): IN: skill registry. Defaults to None. OUT: Consumed during execution.
            llm_client (tp.Any | None, optional): IN: llm client. Defaults to None. OUT: Consumed during execution.
            telemetry (SkillTelemetry | None, optional): IN: telemetry. Defaults to None. OUT: Consumed during execution.
            audit_emitter (AuditEmitter | None, optional): IN: audit emitter. Defaults to None. OUT: Consumed during execution."""
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
        """Reset the tracker for a new agent turn.

        Args:
            agent_id (str | None): IN: Agent identifier. OUT: Passed to
                tracker.
            turn_id (str | None): IN: Turn identifier. OUT: Passed to tracker.
            user_prompt (str): IN: User message text. OUT: Passed to tracker.

        Returns:
            None: OUT: Internal tracker state is reset.
        """

        self.tracker.begin_turn(agent_id=agent_id, turn_id=turn_id, user_prompt=user_prompt)

    def record_call(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        """Forward a tool call event to the internal tracker.

        Args:
            *args: IN: Positional args for ``ToolSequenceTracker.record_call``.
                OUT: Forwarded.
            **kwargs: IN: Keyword args for ``ToolSequenceTracker.record_call``.
                OUT: Forwarded.

        Returns:
            tp.Any: OUT: Return value from the tracker.
        """

        return self.tracker.record_call(*args, **kwargs)

    def on_turn_end(self, final_response: str = "") -> AuthoringResult:
        """Evaluate the tracked turn and draft a skill if criteria are met.

        Args:
            final_response (str): IN: Agent's final text response. OUT: Passed
                to ``end_turn``.

        Returns:
            AuthoringResult: OUT: Detailed result of the authoring attempt.
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
        """Parse the ``name`` field from YAML frontmatter.

        Args:
            text (str): IN: SKILL.md content. OUT: Searched line-by-line.

        Returns:
            str | None: OUT: Name string or ``None``.
        """

        for line in text.splitlines():
            if line.startswith("name:"):
                return line.split(":", 1)[1].strip().strip('"').strip("'")
        return None

    @staticmethod
    def _extract_version(text: str) -> str | None:
        """Parse the ``version`` field from YAML frontmatter.

        Args:
            text (str): IN: SKILL.md content. OUT: Searched line-by-line.

        Returns:
            str | None: OUT: Version string or ``None``.
        """

        for line in text.splitlines():
            if line.startswith("version:"):
                return line.split(":", 1)[1].strip().strip('"').strip("'")
        return None
