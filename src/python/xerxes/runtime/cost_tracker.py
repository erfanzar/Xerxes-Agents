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
"""In-session LLM cost accounting.

Provides :class:`CostEvent` (one billing record) and :class:`CostTracker`
(an append-only ledger held by every :class:`QueryEngine`). Costs are
computed via :func:`xerxes.llms.registry.calc_cost`; prompt-caching reads
and writes are recorded so the ``/cost`` and ``/usage`` slash commands can
report cache-hit rates and discounted spend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class CostEvent:
    """One immutable cost ledger entry.

    Attributes:
        model: LLM identifier the event was billed against.
        in_tokens: Non-cached input tokens.
        out_tokens: Output tokens.
        cost_usd: Dollar cost computed at record time.
        label: Free-form label, typically ``"turn_<n>"``.
        timestamp: ISO-8601 timestamp captured at construction.
        cache_read_tokens: Tokens served from a prompt-cache hit.
        cache_creation_tokens: Tokens written into a new cache entry.
    """

    model: str
    in_tokens: int
    out_tokens: int
    cost_usd: float
    label: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


@dataclass
class CostTracker:
    """Append-only ledger of :class:`CostEvent` entries for one session.

    Attributes:
        events: All cost events recorded so far, in insertion order.
    """

    events: list[CostEvent] = field(default_factory=list)

    def record_turn(
        self,
        model: str,
        in_tokens: int,
        out_tokens: int,
        label: str = "",
        *,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> CostEvent:
        """Record a single LLM turn's token usage and append a priced event.

        Args:
            model: LLM identifier used to look up per-token pricing.
            in_tokens: Non-cached input tokens consumed by the turn.
            out_tokens: Output tokens produced by the turn.
            label: Optional human-friendly label stored on the event.
            cache_read_tokens: Tokens served from a prompt cache; billed at
                ~10% of the normal input rate.
            cache_creation_tokens: Tokens used to populate a new cache entry;
                billed at ~125% of the normal input rate.

        Returns:
            The newly appended :class:`CostEvent`.

        Note:
            ``calc_cost`` does not yet model cache pricing natively; raw
            cache counts are still recorded so the pricing fix can be
            replayed retroactively.
        """

        from xerxes.llms.registry import calc_cost

        # Approximate cache-aware cost: pretend cache_read is 10% of
        # normal input tokens, and cache_creation is 125%.
        effective_in = in_tokens
        cache_extra_cost = 0.0
        if cache_read_tokens or cache_creation_tokens:
            per_in_unit_cost = (calc_cost(model, 1000, 0) / 1000.0) if calc_cost(model, 1000, 0) > 0 else 0.0
            cache_extra_cost = (cache_read_tokens * per_in_unit_cost * 0.1) + (
                cache_creation_tokens * per_in_unit_cost * 1.25
            )
        cost = calc_cost(model, effective_in, out_tokens) + cache_extra_cost
        event = CostEvent(
            model=model,
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            cost_usd=cost,
            label=label,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )
        self.events.append(event)
        return event

    def record_raw(self, label: str, cost_usd: float, model: str = "") -> CostEvent:
        """Record an externally-priced cost (e.g. embeddings, image gen).

        Use this when the caller already knows the dollar amount and has no
        meaningful token counts to attribute.
        """

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
        """Cumulative dollar cost summed across every recorded event."""
        return sum(e.cost_usd for e in self.events)

    @property
    def total_input_tokens(self) -> int:
        """Total non-cached input tokens consumed across every event."""
        return sum(e.in_tokens for e in self.events)

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens produced across every event."""
        return sum(e.out_tokens for e in self.events)

    @property
    def total_tokens(self) -> int:
        """Sum of input and output tokens across every event."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cache_read_tokens(self) -> int:
        """Total tokens served from a prompt-cache hit across every event."""
        return sum(e.cache_read_tokens for e in self.events)

    @property
    def total_cache_creation_tokens(self) -> int:
        """Total tokens written into new cache entries across every event."""
        return sum(e.cache_creation_tokens for e in self.events)

    def cache_hit_rate(self) -> float:
        """Return the fraction of input tokens served from the prompt cache.

        Returns ``0.0`` when no input tokens have been recorded yet.
        """
        served = self.total_cache_read_tokens + self.total_input_tokens
        if served <= 0:
            return 0.0
        return self.total_cache_read_tokens / float(served)

    @property
    def event_count(self) -> int:
        """Number of :class:`CostEvent` records held."""
        return len(self.events)

    def by_model(self) -> dict[str, dict[str, Any]]:
        """Return a per-model aggregate of tokens, cost, and turn count."""

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
        """Drop every recorded :class:`CostEvent`."""
        self.events.clear()

    def summary(self) -> str:
        """Render a Markdown summary of totals plus per-model breakdown."""

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
        """Serialise every event as a JSON-friendly dict for persistence."""

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
