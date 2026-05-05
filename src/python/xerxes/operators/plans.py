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
"""Plans module for Xerxes.

Exports:
    - PlanStateManager"""

from __future__ import annotations

from .types import OperatorPlanState


class PlanStateManager:
    """Plan state manager."""

    def __init__(self) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        self._state = OperatorPlanState()

    @property
    def state(self) -> OperatorPlanState:
        """Return State.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            OperatorPlanState: OUT: Result of the operation."""

        return self._state

    def update(self, explanation: str | None, plan: list[dict[str, str]]) -> dict:
        """Update.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            explanation (str | None): IN: explanation. OUT: Consumed during execution.
            plan (list[dict[str, str]]): IN: plan. OUT: Consumed during execution.
        Returns:
            dict: OUT: Result of the operation."""

        return self._state.update(explanation, plan)

    def summary(self) -> str:
        """Summary.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            str: OUT: Result of the operation."""

        if not self._state.steps:
            return "No plan"
        return ", ".join(f"{step.status}:{step.step}" for step in self._state.steps[:3])
