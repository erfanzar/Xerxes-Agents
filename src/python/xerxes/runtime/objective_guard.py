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
"""Runtime guardrails for objective-mode final answers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .interaction_modes import normalize_interaction_mode

_DEFAULT_OBJECTIVE_GUARD_RETRIES = 6

_VERIFIED_SUCCESS_MARKERS = (
    "acceptance criteria pass",
    "all acceptance criteria pass",
    "objective met",
    "verified complete",
    "verified completion",
    "all tests pass",
    "all benchmarks pass",
    "all checks pass",
)

_UNRESOLVED_MARKERS = (
    "❌",
    "not met",
    "unmet",
    "not yet",
    "still fails",
    "still failing",
    "still fail",
    "still loses",
    "still losing",
    "still lose",
    "cannot beat",
    "can't beat",
    "does not pass",
    "do not pass",
    "failing benchmark",
    "failing test",
    "remaining failure",
    "remaining issue",
    "losses",
    "loses by",
)

_RUNAWAY_FINAL_MARKERS = (
    "want me to",
    "should i",
    "would you like",
    "do you want me",
    "honest final",
    "final state",
    "where we stand",
    "where we are",
    "path forward is",
    "next step is",
)

_BLOCKER_MARKERS = (
    "blocked:",
    "concrete blocker",
    "externally blocked",
    "cannot proceed because",
)

_BLOCKER_EVIDENCE_MARKERS = (
    "evidence:",
    "command:",
    "stderr",
    "traceback",
    "permission denied",
    "not installed",
    "missing dependency",
    "requires user",
)


@dataclass(frozen=True)
class ObjectiveGuardDecision:
    """Decision returned after inspecting an objective-mode no-tool response."""

    should_continue: bool
    reason: str = ""
    reminder: str = ""


def objective_guard_retry_limit(config: dict[str, Any]) -> int:
    """Return the maximum objective guard retries before a visible stop."""
    raw = config.get("objective_guard_max_retries") or os.environ.get("XERXES_OBJECTIVE_GUARD_MAX_RETRIES")
    try:
        return max(1, int(raw)) if raw is not None else _DEFAULT_OBJECTIVE_GUARD_RETRIES
    except (TypeError, ValueError):
        return _DEFAULT_OBJECTIVE_GUARD_RETRIES


def inspect_objective_response(text: str, *, mode: Any) -> ObjectiveGuardDecision:
    """Reject objective-mode no-tool responses that do not prove completion.

    Objective mode is a runtime contract, not just a prompt hint. A text-only
    assistant response can end the turn only when it clearly states verified
    success or a concrete blocker with evidence.
    """
    if normalize_interaction_mode(mode) != "objective":
        return ObjectiveGuardDecision(should_continue=False)

    stripped = text.strip()
    if not stripped:
        return ObjectiveGuardDecision(should_continue=False)

    lowered = stripped.lower()
    success_marker = _first_marker(lowered, _VERIFIED_SUCCESS_MARKERS)
    unresolved_marker = _first_marker(lowered, _UNRESOLVED_MARKERS)
    if success_marker and not unresolved_marker:
        return ObjectiveGuardDecision(should_continue=False)

    blocker_marker = _first_marker(lowered, _BLOCKER_MARKERS)
    blocker_evidence = _first_marker(lowered, _BLOCKER_EVIDENCE_MARKERS)
    if blocker_marker and blocker_evidence:
        return ObjectiveGuardDecision(should_continue=False)

    runaway_marker = _first_marker(lowered, _RUNAWAY_FINAL_MARKERS)
    reason = (
        f"unresolved acceptance marker `{unresolved_marker}`"
        if unresolved_marker
        else f"premature stopping marker `{runaway_marker}`"
        if runaway_marker
        else "no verified completion or concrete blocker evidence"
    )
    return ObjectiveGuardDecision(
        should_continue=True,
        reason=reason,
        reminder=(
            "[Objective gate]\n"
            f"The previous assistant response tried to stop, but objective mode is still active: {reason}.\n"
            "Continue the hard-goal loop. Do not final-answer with a narrative status. Update the ledger, "
            "choose the next concrete hypothesis, use tools to edit or verify, and only end after all acceptance "
            "criteria pass or after you report `BLOCKED:` with exact evidence."
        ),
    )


def _first_marker(text: str, markers: tuple[str, ...]) -> str:
    """Return the first marker contained in ``text``."""
    for marker in markers:
        if marker in text:
            return marker
    return ""


__all__ = [
    "ObjectiveGuardDecision",
    "inspect_objective_response",
    "objective_guard_retry_limit",
]
