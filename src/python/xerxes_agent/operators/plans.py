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

"""Plan-state manager for operator tooling.

Provides :class:`PlanStateManager`, a thin wrapper around
:class:`~xerxes_agent.operators.types.OperatorPlanState` that exposes
convenience methods for updating and summarising the current execution
plan.
"""

from __future__ import annotations

from .types import OperatorPlanState


class PlanStateManager:
    """Manage per-runtime operator plan state.

    Owns a single :class:`OperatorPlanState` instance and provides
    methods to replace its contents and generate compact summaries
    suitable for the TUI status bar.

    Attributes:
        _state: The underlying :class:`OperatorPlanState` that holds
            the current plan steps, explanation, and revision counter.
    """

    def __init__(self) -> None:
        """Initialise the manager with an empty plan state."""
        self._state = OperatorPlanState()

    @property
    def state(self) -> OperatorPlanState:
        """Return the current plan state.

        Returns:
            The underlying :class:`OperatorPlanState` instance.
        """
        return self._state

    def update(self, explanation: str | None, plan: list[dict[str, str]]) -> dict:
        """Replace the current plan contents.

        Delegates to :meth:`OperatorPlanState.update` to atomically
        swap the explanation and step list.

        Args:
            explanation: Optional short note describing the plan change
                or current situation.
            plan: List of step dictionaries.  Each dictionary should
                contain at least a ``"step"`` key and an optional
                ``"status"`` key (defaults to ``"pending"``).

        Returns:
            The serialised plan state dictionary produced by
            :meth:`OperatorPlanState.to_dict`.
        """
        return self._state.update(explanation, plan)

    def summary(self) -> str:
        """Return a compact plan summary string for the TUI.

        Produces a comma-separated list of the first three steps in
        ``status:step`` format.

        Returns:
            A human-readable summary string.  Returns ``"No plan"``
            when no steps have been recorded.
        """
        if not self._state.steps:
            return "No plan"
        return ", ".join(f"{step.status}:{step.step}" for step in self._state.steps[:3])
