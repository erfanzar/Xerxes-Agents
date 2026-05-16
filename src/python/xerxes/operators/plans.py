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
"""Session-scoped owner of the operator's structured execution plan.

Wraps a single :class:`OperatorPlanState` so the runtime can hold a stable
reference even when the plan is replaced wholesale by the ``update_plan``
tool.
"""

from __future__ import annotations

from .types import OperatorPlanState


class PlanStateManager:
    """Stateful holder around one :class:`OperatorPlanState` instance.

    The manager exists so that the surrounding session has a stable object
    to point at — the underlying plan state can be mutated in-place by
    :meth:`update` without invalidating cached references.
    """

    def __init__(self) -> None:
        """Allocate an empty plan state."""

        self._state = OperatorPlanState()

    @property
    def state(self) -> OperatorPlanState:
        """The live plan state owned by this manager."""

        return self._state

    def update(self, explanation: str | None, plan: list[dict[str, str]]) -> dict:
        """Replace the plan contents and return the new wire snapshot.

        Args:
            explanation: New plan preamble (``None`` clears it).
            plan: Ordered list of ``{"step": ..., "status": ...}`` dicts.
        """

        return self._state.update(explanation, plan)

    def summary(self) -> str:
        """Return a short status string for the first few plan steps."""

        if not self._state.steps:
            return "No plan"
        return ", ".join(f"{step.status}:{step.step}" for step in self._state.steps[:3])
