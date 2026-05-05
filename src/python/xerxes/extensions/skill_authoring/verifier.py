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
"""Skill verification recipe generation and evaluation.

``SkillVerifier`` produces ``VerificationStep`` objects from a
``SkillCandidate`` and can later validate whether another candidate satisfies
those steps.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

from .tracker import SkillCandidate, ToolCallEvent


@dataclass
class VerificationStep:
    """A single assertion in a skill verification recipe.

    Attributes:
        kind (str): IN: Step type (``call_count``, ``sequence_position``,
            ``args_subset``, etc.). OUT: Determines evaluation logic.
        tool_name (str): IN: Target tool. OUT: Used for sequence and args
            checks.
        position (int | None): IN: Expected index in the event list. OUT:
            Used for positional checks.
        args_required (dict[str, tp.Any]): IN: Required argument keys/values.
            OUT: Checked during ``args_subset`` evaluation.
        expected_count (int | None): IN: Expected number of successful calls.
            OUT: Checked during ``call_count`` evaluation.
        message (str): IN: Human-readable description. OUT: Stored.
    """

    kind: str
    tool_name: str = ""
    position: int | None = None
    args_required: dict[str, tp.Any] = field(default_factory=dict)
    expected_count: int | None = None
    message: str = ""


@dataclass
class VerificationResult:
    """Outcome of verifying a candidate against a recipe.

    Attributes:
        passed (bool): IN: Overall result. OUT: ``True`` if ``failed_steps``
            is empty.
        passed_steps (list[int]): IN: Empty initially. OUT: Indices of steps
            that passed.
        failed_steps (list[tuple[int, str]]): IN: Empty initially. OUT:
            Indices and reasons for steps that failed.
    """

    passed: bool
    passed_steps: list[int] = field(default_factory=list)
    failed_steps: list[tuple[int, str]] = field(default_factory=list)


class SkillVerifier:
    """Generate and evaluate skill verification recipes."""

    def generate(self, candidate: SkillCandidate) -> list[VerificationStep]:
        """Build a verification recipe from a successful tool sequence.

        Args:
            candidate (SkillCandidate): IN: Observed turn data. OUT: Successful
                events are turned into steps.

        Returns:
            list[VerificationStep]: OUT: Recipe steps.
        """

        steps: list[VerificationStep] = []
        successful = candidate.successful_events
        steps.append(
            VerificationStep(
                kind="call_count",
                expected_count=len(successful),
                message=f"expects {len(successful)} successful tool calls",
            )
        )
        for i, ev in enumerate(successful):
            steps.append(
                VerificationStep(
                    kind="sequence_position",
                    tool_name=ev.tool_name,
                    position=i,
                    message=f"position {i} should call {ev.tool_name}",
                )
            )
            if ev.arguments:
                steps.append(
                    VerificationStep(
                        kind="args_subset",
                        tool_name=ev.tool_name,
                        position=i,
                        args_required={k: ev.arguments[k] for k in list(ev.arguments)[:3]},
                        message=f"{ev.tool_name} expects keys {list(ev.arguments)[:3]}",
                    )
                )
        return steps

    def verify(
        self,
        steps: list[VerificationStep],
        candidate: SkillCandidate,
    ) -> VerificationResult:
        """Evaluate whether ``candidate`` satisfies every recipe step.

        Args:
            steps (list[VerificationStep]): IN: Recipe to evaluate. OUT:
                Iterated and passed to ``_evaluate``.
            candidate (SkillCandidate): IN: Observed turn to validate. OUT:
                Successful events are extracted.

        Returns:
            VerificationResult: OUT: Aggregated pass/fail result.
        """

        observed = candidate.successful_events
        passed: list[int] = []
        failed: list[tuple[int, str]] = []
        for i, step in enumerate(steps):
            ok, reason = self._evaluate(step, observed)
            if ok:
                passed.append(i)
            else:
                failed.append((i, reason))
        return VerificationResult(passed=not failed, passed_steps=passed, failed_steps=failed)

    def _evaluate(
        self,
        step: VerificationStep,
        observed: list[ToolCallEvent],
    ) -> tuple[bool, str]:
        """Check a single verification step against observed events.

        Args:
            step (VerificationStep): IN: Step to evaluate. OUT: ``kind``
                selects the logic branch.
            observed (list[ToolCallEvent]): IN: Successful events from the
                candidate. OUT: Compared against the step.

        Returns:
            tuple[bool, str]: OUT: ``(True, "")`` on pass, or
            ``(False, reason)`` on failure.
        """

        if step.kind == "call_count":
            if step.expected_count is not None and len(observed) != step.expected_count:
                return False, f"expected {step.expected_count} successful calls, got {len(observed)}"
            return True, ""
        if step.kind == "tool_called":
            if not any(e.tool_name == step.tool_name for e in observed):
                return False, f"tool {step.tool_name!r} was never called"
            return True, ""
        if step.kind == "sequence_position":
            if step.position is None or step.position >= len(observed):
                return False, f"position {step.position} not in observed sequence"
            actual = observed[step.position].tool_name
            if actual != step.tool_name:
                return False, f"expected {step.tool_name!r} at pos {step.position}, got {actual!r}"
            return True, ""
        if step.kind == "args_subset":
            if step.position is None or step.position >= len(observed):
                return False, "position out of range"
            actual_args = observed[step.position].arguments
            for k in step.args_required:
                if k not in actual_args:
                    return False, f"missing required arg key {k!r}"
            return True, ""
        if step.kind == "status_success":
            if step.position is None or step.position >= len(observed):
                return False, "position out of range"
            return (observed[step.position].status == "success"), "non-success status"
        return False, f"unknown step kind {step.kind!r}"
