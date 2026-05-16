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
"""Analytics over recent :class:`CostEvent` records for the ``/insights`` command.

Aggregates a stream of :class:`CostEvent` entries (typically from a
:class:`CostTracker`) into daily totals, per-model and per-label breakdowns,
projected monthly burn, and cache savings. Tests inject synthetic events so
no live :class:`CostTracker` is needed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from .cost_tracker import CostEvent


def _parse_dt(s: str) -> datetime:
    """Parse ISO-8601 ``s`` to ``datetime``, falling back to now-UTC on failure."""
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.now(UTC)


@dataclass
class InsightsReport:
    """Aggregated stats over a list of :class:`CostEvent` records.

    Attributes:
        total_events: Number of events that contributed to this report.
        total_cost_usd: Sum of ``cost_usd`` across events.
        total_input_tokens: Sum of non-cached input tokens.
        total_output_tokens: Sum of output tokens.
        total_cache_read_tokens: Sum of cache-hit input tokens.
        total_cache_creation_tokens: Sum of cache-write tokens.
        by_model: Per-model rollup with events/cost/in/out token totals.
        by_day: Per-day rollup with event count and cost.
        by_label: Event-count histogram by ``label``.
        cache_hit_rate: Fraction of input tokens served from the cache.
        projected_monthly_cost: ``avg_daily_cost * 30`` extrapolation.
    """

    total_events: int = 0
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)
    by_day: dict[str, dict[str, float]] = field(default_factory=dict)
    by_label: dict[str, int] = field(default_factory=dict)
    cache_hit_rate: float = 0.0
    projected_monthly_cost: float = 0.0


def build_report(events: Iterable[CostEvent], *, days: int | None = None, now: datetime | None = None) -> InsightsReport:
    """Build an :class:`InsightsReport` from ``events``, optionally clipped to the last ``days``.

    Args:
        events: Iterable of :class:`CostEvent` records.
        days: When set, only events from the last ``days`` are included.
        now: Override clock for ``days`` filtering (defaults to UTC now).
    """
    horizon = (now or datetime.now(UTC)) - timedelta(days=days) if days else None
    rpt = InsightsReport()
    for ev in events:
        ts = _parse_dt(ev.timestamp)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        if horizon is not None and ts < horizon:
            continue
        rpt.total_events += 1
        rpt.total_cost_usd += ev.cost_usd
        rpt.total_input_tokens += ev.in_tokens
        rpt.total_output_tokens += ev.out_tokens
        rpt.total_cache_read_tokens += ev.cache_read_tokens
        rpt.total_cache_creation_tokens += ev.cache_creation_tokens
        model_bucket = rpt.by_model.setdefault(ev.model, {"events": 0, "cost_usd": 0.0, "in_tokens": 0, "out_tokens": 0})
        model_bucket["events"] += 1
        model_bucket["cost_usd"] += ev.cost_usd
        model_bucket["in_tokens"] += ev.in_tokens
        model_bucket["out_tokens"] += ev.out_tokens
        day_key = ts.date().isoformat()
        day_bucket = rpt.by_day.setdefault(day_key, {"events": 0, "cost_usd": 0.0})
        day_bucket["events"] += 1
        day_bucket["cost_usd"] += ev.cost_usd
        if ev.label:
            rpt.by_label[ev.label] = rpt.by_label.get(ev.label, 0) + 1

    served = rpt.total_input_tokens + rpt.total_cache_read_tokens
    rpt.cache_hit_rate = rpt.total_cache_read_tokens / served if served else 0.0
    if rpt.by_day:
        avg_daily = rpt.total_cost_usd / max(len(rpt.by_day), 1)
        rpt.projected_monthly_cost = avg_daily * 30.0
    return rpt


def format_report(rpt: InsightsReport) -> str:
    """Render an :class:`InsightsReport` as a Markdown summary."""
    lines = [
        "# Xerxes Insights",
        "",
        f"Events: {rpt.total_events}",
        f"Total cost: ${rpt.total_cost_usd:.4f}",
        f"Input tokens: {rpt.total_input_tokens:,}",
        f"Output tokens: {rpt.total_output_tokens:,}",
        f"Cache read tokens: {rpt.total_cache_read_tokens:,}",
        f"Cache hit rate: {rpt.cache_hit_rate * 100:.1f}%",
        f"Projected monthly cost: ${rpt.projected_monthly_cost:.2f}",
    ]
    if rpt.by_model:
        lines.append("\n## Top models")
        for model, stats in sorted(rpt.by_model.items(), key=lambda kv: -kv[1]["cost_usd"]):
            lines.append(
                f"- {model}: ${stats['cost_usd']:.4f} ({stats['events']:.0f} events, "
                f"{int(stats['in_tokens']):,} in / {int(stats['out_tokens']):,} out)"
            )
    return "\n".join(lines)


__all__ = ["InsightsReport", "build_report", "format_report"]
