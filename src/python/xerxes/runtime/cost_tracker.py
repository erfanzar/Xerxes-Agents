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


"""Granular cost tracking for LLM API usage.

Tracks per-event, per-model, and per-session costs with support for
token-level and dollar-level accounting.

Inspired by the claw-code ``CostTracker`` and enhanced with Xerxes's
provider registry cost tables.

Usage::

    from xerxes.runtime.cost_tracker import CostTracker

    tracker = CostTracker()
    tracker.record_turn("gpt-4o", in_tokens=1500, out_tokens=800)
    tracker.record_turn("claude-opus-4-6", in_tokens=2000, out_tokens=500)
    print(tracker.total_cost_usd)
    print(tracker.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class CostEvent:
    """A single cost event.

    Attributes:
        model: Model name used.
        in_tokens: Input tokens consumed.
        out_tokens: Output tokens generated.
        cost_usd: Estimated cost in USD.
        label: Human-readable label (e.g. ``"turn_3"``, ``"tool_agent"``).
        timestamp: ISO 8601 timestamp.
    """

    model: str
    in_tokens: int
    out_tokens: int
    cost_usd: float
    label: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CostTracker:
    """Tracks API costs across an entire session.

    Provides per-event recording, per-model aggregation, and
    formatted summaries.
    """

    events: list[CostEvent] = field(default_factory=list)

    def record_turn(
        self,
        model: str,
        in_tokens: int,
        out_tokens: int,
        label: str = "",
    ) -> CostEvent:
        """Record a single LLM turn.

        Calculates cost using the provider registry's pricing table.

        Args:
            model: Model name.
            in_tokens: Input tokens for this turn.
            out_tokens: Output tokens for this turn.
            label: Optional label for this event.

        Returns:
            The created cost event.
        """
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
        """Record a raw cost event (no token calculation)."""
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
        """Total cost across all events."""
        return sum(e.cost_usd for e in self.events)

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all events."""
        return sum(e.in_tokens for e in self.events)

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all events."""
        return sum(e.out_tokens for e in self.events)

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def event_count(self) -> int:
        return len(self.events)

    def by_model(self) -> dict[str, dict[str, Any]]:
        """Aggregate costs by model.

        Returns:
            Dict mapping model name to aggregated stats.
        """
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
        self.events.clear()

    def summary(self) -> str:
        """Return a formatted cost summary."""
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
        """Serialize events for JSON storage."""
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
