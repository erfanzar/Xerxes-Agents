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
"""Thread-safe optional iteration budget for the agent loop.

``max_iterations=None`` means unbounded: the loop relies on cancellation,
context-window provisioning, and explicit token budgets instead of a hidden
fixed tool-turn cap. A positive value re-enables a hard ceiling.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any


class BudgetExhausted(RuntimeError):
    """Raised when an attempt is made to consume from an exhausted budget."""


@dataclass
class IterationBudget:
    """Thread-safe iteration counter with refundable consumption.

    Use :meth:`consume` to spend a turn and inspect :attr:`remaining`. Call
    :meth:`refund` after a programmatic tool call so that one LLM turn that
    expanded into many internal invocations counts as a single iteration.
    Direct mutation of ``_used`` is prevented; the counter only advances or
    retracts via :meth:`consume` and :meth:`refund`.

    Attributes:
        max_iterations: Optional hard ceiling on accumulated iterations.
            ``None`` means no iteration cap.
    """

    max_iterations: int | None = None
    _used: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize non-positive limits to uncapped."""

        if self.max_iterations is not None and self.max_iterations <= 0:
            self.max_iterations = None

    @property
    def used(self) -> int:
        """Number of iterations currently charged to the budget."""
        with self._lock:
            return self._used

    @property
    def remaining(self) -> int | None:
        """Iterations still available, or ``None`` when unbounded."""
        with self._lock:
            if self.max_iterations is None:
                return None
            return max(0, self.max_iterations - self._used)

    @property
    def exhausted(self) -> bool:
        """``True`` when ``used >= max_iterations``."""
        with self._lock:
            if self.max_iterations is None:
                return False
            return self._used >= self.max_iterations

    def consume(self, n: int = 1) -> int:
        """Charge ``n`` iterations to the budget and return the new total.

        Args:
            n: Number of iterations to consume; must be positive.

        Raises:
            ValueError: ``n`` is non-positive.
            BudgetExhausted: Charging ``n`` would exceed ``max_iterations``.
        """
        if n <= 0:
            raise ValueError("n must be positive")
        with self._lock:
            if self.max_iterations is not None and self._used + n > self.max_iterations:
                raise BudgetExhausted(
                    f"Iteration budget exhausted (used={self._used}, max={self.max_iterations}, asked={n})"
                )
            self._used += n
            return self._used

    def try_consume(self, n: int = 1) -> bool:
        """Best-effort :meth:`consume`; returns ``True`` on success, ``False`` otherwise."""
        try:
            self.consume(n)
            return True
        except BudgetExhausted:
            return False

    def refund(self, n: int = 1) -> int:
        """Refund up to ``n`` iterations; ``used`` never drops below zero.

        Raises:
            ValueError: ``n`` is non-positive.
        """
        if n <= 0:
            raise ValueError("n must be positive")
        with self._lock:
            self._used = max(0, self._used - n)
            return self._used

    def reset(self) -> None:
        """Restore the budget to a fresh zero-used state."""
        with self._lock:
            self._used = 0


def iteration_budget_from_config(
    config: dict[str, Any],
    *,
    key: str = "max_tool_turns",
    env_var: str = "XERXES_MAX_TOOL_TURNS",
) -> IterationBudget:
    """Build an optional budget from runtime config and environment.

    A missing, empty, zero, or negative value means uncapped.
    """

    raw: Any = config.get(key)
    if raw in (None, ""):
        raw = os.environ.get(env_var)
    if raw in (None, ""):
        return IterationBudget()
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return IterationBudget()
    return IterationBudget(max_iterations=parsed if parsed > 0 else None)


__all__ = ["BudgetExhausted", "IterationBudget", "iteration_budget_from_config"]
