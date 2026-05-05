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
"""Cost tracker module for Xerxes.

Exports:
    - CostEvent
    - CostTracker"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class CostEvent:
    """Cost event.

    Attributes:
        model (str): model.
        in_tokens (int): in tokens.
        out_tokens (int): out tokens.
        cost_usd (float): cost usd.
        label (str): label.
        timestamp (str): timestamp."""

    model: str
    in_tokens: int
    out_tokens: int
    cost_usd: float
    label: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CostTracker:
    """Cost tracker.

    Attributes:
        events (list[CostEvent]): events."""

    events: list[CostEvent] = field(default_factory=list)

    def record_turn(
        self,
        model: str,
        in_tokens: int,
        out_tokens: int,
        label: str = "",
    ) -> CostEvent:
        """Record turn.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            model (str): IN: model. OUT: Consumed during execution.
            in_tokens (int): IN: in tokens. OUT: Consumed during execution.
            out_tokens (int): IN: out tokens. OUT: Consumed during execution.
            label (str, optional): IN: label. Defaults to ''. OUT: Consumed during execution.
        Returns:
            CostEvent: OUT: Result of the operation."""

        from xerxes.llms.registry import calc_cost

        cost = calc_cost(model, in_tokens, out_tokens)
        event = CostEvent(
            model=model,
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            cost_usd=cost,
            label=label,
        )
        self.events.append(event)
        return event

    def record_raw(self, label: str, cost_usd: float, model: str = "") -> CostEvent:
        """Record raw.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            label (str): IN: label. OUT: Consumed during execution.
            cost_usd (float): IN: cost usd. OUT: Consumed during execution.
            model (str, optional): IN: model. Defaults to ''. OUT: Consumed during execution.
        Returns:
            CostEvent: OUT: Result of the operation."""

        event = CostEvent(
            model=model,
            in_tokens=0,
            out_tokens=0,
            cost_usd=cost_usd,
            label=label,
        )
        self.events.append(event)
        return event

    @property
    def total_cost_usd(self) -> float:
        """Return Total cost usd.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            float: OUT: Result of the operation."""

        return sum(e.cost_usd for e in self.events)

    @property
    def total_input_tokens(self) -> int:
        """Return Total input tokens.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            int: OUT: Result of the operation."""

        return sum(e.in_tokens for e in self.events)

    @property
    def total_output_tokens(self) -> int:
        """Return Total output tokens.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            int: OUT: Result of the operation."""

        return sum(e.out_tokens for e in self.events)

    @property
    def total_tokens(self) -> int:
        """Return Total tokens.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            int: OUT: Result of the operation."""

        return self.total_input_tokens + self.total_output_tokens

    @property
    def event_count(self) -> int:
        """Return Event count.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            int: OUT: Result of the operation."""
        return len(self.events)

    def by_model(self) -> dict[str, dict[str, Any]]:
        """By model.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, dict[str, Any]]: OUT: Result of the operation."""

        agg: dict[str, dict[str, Any]] = {}
        for e in self.events:
            if e.model not in agg:
                agg[e.model] = {"in_tokens": 0, "out_tokens": 0, "cost_usd": 0.0, "turns": 0}
            agg[e.model]["in_tokens"] += e.in_tokens
            agg[e.model]["out_tokens"] += e.out_tokens
            agg[e.model]["cost_usd"] += e.cost_usd
            agg[e.model]["turns"] += 1
        return agg

    def clear(self) -> None:
        """Clear.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""
        self.events.clear()

    def summary(self) -> str:
        """Summary.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            str: OUT: Result of the operation."""

        lines = [
            "# Cost Summary",
            "",
            f"Total cost: ${self.total_cost_usd:.4f}",
            f"Total tokens: {self.total_tokens:,} (in: {self.total_input_tokens:,}, out: {self.total_output_tokens:,})",
            f"Events: {self.event_count}",
            "",
        ]

        by_model = self.by_model()
        if by_model:
            lines.append("## By Model")
            for model, stats in sorted(by_model.items()):
                lines.append(
                    f"- **{model}**: ${stats['cost_usd']:.4f} "
                    f"({stats['turns']} turns, {stats['in_tokens'] + stats['out_tokens']:,} tokens)"
                )

        return "\n".join(lines)

    def as_dicts(self) -> list[dict[str, Any]]:
        """As dicts.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[dict[str, Any]]: OUT: Result of the operation."""

        return [
            {
                "model": e.model,
                "in_tokens": e.in_tokens,
                "out_tokens": e.out_tokens,
                "cost_usd": e.cost_usd,
                "label": e.label,
                "timestamp": e.timestamp,
            }
            for e in self.events
        ]


__all__ = [
    "CostEvent",
    "CostTracker",
]
