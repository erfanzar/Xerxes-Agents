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
"""Shared interaction-mode vocabulary for TUI, bridge, daemon, and tools."""

from __future__ import annotations

from typing import Any

INTERACTION_MODES = frozenset({"code", "researcher", "plan", "objective"})

MODE_ALIASES = {
    "": "code",
    "coding": "code",
    "coder": "code",
    "code": "code",
    "research": "researcher",
    "researcher": "researcher",
    "plan": "plan",
    "planner": "plan",
    "goal": "objective",
    "goals": "objective",
    "goal-runner": "objective",
    "goal_runner": "objective",
    "objective": "objective",
    "objectives": "objective",
    "iterate": "objective",
    "autonomous": "objective",
}


def normalize_interaction_mode(mode: Any, *, plan_mode: bool = False) -> str:
    """Coerce user/model mode labels to Xerxes' canonical interaction modes."""
    if plan_mode:
        return "plan"
    return MODE_ALIASES.get(str(mode or "code").strip().lower(), "code")


def agent_name_for_mode(mode: str) -> str:
    """Map an interaction mode to the matching agent definition name."""
    normalized = normalize_interaction_mode(mode)
    if normalized == "plan":
        return "planner"
    if normalized == "researcher":
        return "researcher"
    if normalized == "objective":
        return "objective"
    return "coder"


def mode_switch_hint(mode: str) -> str:
    """Return model-facing instructions for switching between interaction modes."""
    normalized = normalize_interaction_mode(mode)
    if normalized == "plan":
        return (
            "[Mode control]\n"
            "You are in plan mode. Produce a plan only. If implementation should begin in a later turn, "
            'call SetInteractionModeTool(mode="code"). If the user gave measurable acceptance criteria and '
            'expects iterative implementation until they pass, call SetInteractionModeTool(mode="objective").'
        )
    if normalized == "researcher":
        return (
            "[Mode control]\n"
            "You are in researcher mode. Gather evidence and answer with citations. If implementation is needed "
            'after your findings, call SetInteractionModeTool(mode="code"). If the task needs repeated '
            'change/verify iterations against acceptance criteria, call SetInteractionModeTool(mode="objective").'
        )
    if normalized == "objective":
        return (
            "[Mode control]\n"
            "You are in objective mode. Treat the user's requested outcome as a hard objective with acceptance "
            "criteria. Maintain a compact task ledger, choose one hypothesis at a time, edit/build/test/benchmark, "
            "compare results to the acceptance criteria, keep or revert based on evidence, and continue. Do not "
            "final-answer with a narrative status while the acceptance criteria are unmet. Leave objective mode "
            "only after verification proves the objective is met, the user changes modes, or you are concretely "
            "blocked and can name the blocker plus the exact evidence. For pure research call "
            'SetInteractionModeTool(mode="researcher"); for design-only work call SetInteractionModeTool(mode="plan"); '
            'after verified completion call SetInteractionModeTool(mode="code").'
        )
    return (
        "[Mode control]\n"
        "Use code mode for normal implementation. If this task should first be researched or planned, call "
        'SetInteractionModeTool(mode="researcher") or SetInteractionModeTool(mode="plan"). If the user asks for '
        "a measurable outcome that requires repeated attempts until tests, benchmarks, or checks pass, call "
        'SetInteractionModeTool(mode="objective").'
    )


__all__ = [
    "INTERACTION_MODES",
    "MODE_ALIASES",
    "agent_name_for_mode",
    "mode_switch_hint",
    "normalize_interaction_mode",
]
