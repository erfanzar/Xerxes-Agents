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
"""Rich status-bar snapshot generator.

Produces a per-turn snapshot that the ``FooterRenderer`` can format.
Fields covered:
    * model
    * input / output / cache_read / cache_write tokens
    * cost (USD)
    * context window utilization (`%`)
    * turn duration (mm:ss)

The snapshot is a pure dataclass, so it's trivial to unit-test and to
wire into either the existing footer or the upcoming context-bar
component."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class StatusSnapshot:
    """Everything the status bar can render in a single turn, as one record.

    Attributes:
        model: Active model identifier, e.g. ``"claude-opus-4-7"``.
        input_tokens: Fresh (non-cache) prompt tokens used this turn.
        output_tokens: Completion tokens generated this turn.
        cache_read_tokens: Prompt tokens served from the provider cache.
        cache_write_tokens: Tokens written to the cache this turn.
        context_window: Hard model context limit (used for the % meter).
        cost_usd: Running cost in USD.
        duration_sec: Wall-clock duration since turn start.
        permission_mode: ``"auto"`` / ``"manual"`` / ``"accept-all"``.
        queue_depth: Pending user prompts waiting behind the current turn.
        active_skill: Skill name when a skill is currently driving the agent.
    """

    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    context_window: int = 200_000
    cost_usd: float = 0.0
    duration_sec: float = 0.0
    permission_mode: str = "auto"
    queue_depth: int = 0
    active_skill: str = ""

    @property
    def context_used(self) -> int:
        """Effective context tokens (fresh input + cache reads)."""
        return self.input_tokens + self.cache_read_tokens

    @property
    def context_percent(self) -> float:
        """Context utilization as a 0-100 float, capped at 100."""
        if self.context_window <= 0:
            return 0.0
        return min(100.0, 100.0 * self.context_used / self.context_window)

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of prompt tokens served from cache (0.0-1.0)."""
        total = self.input_tokens + self.cache_read_tokens
        if total <= 0:
            return 0.0
        return self.cache_read_tokens / total

    def to_dict(self) -> dict[str, float | int | str]:
        """Return ``asdict(self)`` augmented with derived percent / hit-rate."""
        d = asdict(self)
        d["context_percent"] = self.context_percent
        d["cache_hit_rate"] = self.cache_hit_rate
        return d


def format_status(snapshot: StatusSnapshot) -> str:
    """Render a single-line status string.

    Layout::

        model | in/out/cache | $0.0123 | 12% ctx | 00:42
    """
    cache_part = ""
    if snapshot.cache_read_tokens or snapshot.cache_write_tokens:
        cache_part = f"/{_compact(snapshot.cache_read_tokens)}c/{_compact(snapshot.cache_write_tokens)}cw"
    duration = _mmss(snapshot.duration_sec)
    extras = []
    if snapshot.queue_depth:
        extras.append(f"queued={snapshot.queue_depth}")
    if snapshot.active_skill:
        extras.append(f"skill={snapshot.active_skill}")
    if snapshot.permission_mode and snapshot.permission_mode != "auto":
        extras.append(snapshot.permission_mode)
    extras_str = (" | " + " ".join(extras)) if extras else ""
    return (
        f"{snapshot.model or '(no model)'} | "
        f"{_compact(snapshot.input_tokens)}in/{_compact(snapshot.output_tokens)}out{cache_part} | "
        f"${snapshot.cost_usd:.4f} | "
        f"{snapshot.context_percent:.0f}% ctx | "
        f"{duration}"
        f"{extras_str}"
    )


def _compact(n: int) -> str:
    """Format ``n`` with ``K`` / ``M`` suffix for compact display."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _mmss(seconds: float) -> str:
    """Format ``seconds`` as zero-padded ``MM:SS``."""
    sec = max(0, int(seconds))
    m, s = divmod(sec, 60)
    return f"{m:02d}:{s:02d}"


__all__ = ["StatusSnapshot", "format_status"]
