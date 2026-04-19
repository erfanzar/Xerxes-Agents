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

"""Skill verifier — generate assertable verification recipes.

A drafted skill is most useful when it ships with a programmatic check
that a future agent's invocation actually followed the recipe. This
module produces an assertion list (one per call) that the runtime can
later replay against a candidate sequence to decide whether the agent
"applied the skill correctly".
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

from .tracker import SkillCandidate, ToolCallEvent


@dataclass
class VerificationStep:
    """One assertable expectation in a skill verification recipe.

    Attributes:
        kind: Assertion kind. One of ``"tool_called"``, ``"args_subset"``,
            ``"sequence_position"``, ``"call_count"``, ``"status_success"``.
        tool_name: Tool the assertion concerns.
        position: Expected 0-indexed position in the sequence (when
            ``kind == "sequence_position"``).
        args_required: Required argument keys/values (when
            ``kind == "args_subset"``).
        message: Human-readable description.
    """

    kind: str
    tool_name: str = ""
    position: int | None = None
    args_required: dict[str, tp.Any] = field(default_factory=dict)
    expected_count: int | None = None
    message: str = ""


@dataclass
class VerificationResult:
    """Outcome of running a recipe against an actual sequence.

    Attributes:
        passed: Whether all assertions held.
        passed_steps: Indices of steps that passed.
        failed_steps: List of ``(index, reason)`` tuples for failures.
    """

    passed: bool
    passed_steps: list[int] = field(default_factory=list)
    failed_steps: list[tuple[int, str]] = field(default_factory=list)


class SkillVerifier:
    """Generate and run skill verification recipes.

    Use :meth:`generate` at draft time to produce a recipe that ships
    with the SKILL.md (e.g. as a JSON sidecar). Use :meth:`verify` to
    evaluate a future :class:`SkillCandidate` against that recipe.
    """

    def generate(self, candidate: SkillCandidate) -> list[VerificationStep]:
        """Produce a verification recipe for *candidate*.

        Recipe shape:

        - ``call_count`` step: total successful calls expected.
        - For each successful call, a ``sequence_position`` step
          asserting the tool name at that index.
        - For calls with non-trivial arguments, an ``args_subset`` step
          asserting that future arguments include the same keys.

        Args:
            candidate: Source candidate to derive the recipe from.

        Returns:
            Ordered list of :class:`VerificationStep` instances.
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
        """Run a recipe against an actual sequence.

        Args:
            steps: The generated recipe.
            candidate: Real sequence captured at runtime.

        Returns:
            A :class:`VerificationResult` summarising the outcome.
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
        """Evaluate a single verification step against the observed events."""
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
