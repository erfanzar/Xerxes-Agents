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
"""Per-skill telemetry aggregator.

Subscribes to :class:`SkillUsedEvent` / :class:`SkillFeedbackEvent` /
:class:`SkillAuthoredEvent` and maintains rolling stats per skill:

- Invocation count and success rate.
- p50/p95 latency.
- Last-failure reason and timestamp.
- Net feedback score.

Powers Epic 7's auto-deprecation and A/B testing logic.
"""

from __future__ import annotations

import bisect
import threading
import typing as tp
from dataclasses import dataclass, field
from datetime import datetime

from ...audit.events import (
    SkillAuthoredEvent,
    SkillFeedbackEvent,
    SkillUsedEvent,
)


@dataclass
class SkillStats:
    """Rolling per-skill statistics.

    Attributes:
        skill_name: The skill being tracked.
        version: Latest version observed.
        invocations: Total :class:`SkillUsedEvent` count.
        successes: Successful invocations.
        failures: Failed invocations.
        durations_ms: Sorted list of invocation latencies.
        last_invoked: Timestamp of the most recent invocation.
        last_failure_reason: ``outcome`` field of the most recent failure.
        feedback_good: Count of ``"good"`` feedback events.
        feedback_bad: Count of ``"bad"`` feedback events.
        authored_at: Timestamp from the originating ``SkillAuthoredEvent``.
    """

    skill_name: str
    version: str = ""
    invocations: int = 0
    successes: int = 0
    failures: int = 0
    durations_ms: list[float] = field(default_factory=list)
    last_invoked: datetime | None = None
    last_failure_reason: str = ""
    feedback_good: int = 0
    feedback_bad: int = 0
    authored_at: datetime | None = None

    @property
    def success_rate(self) -> float:
        """Successes ÷ invocations (0.0 when never invoked)."""
        if self.invocations == 0:
            return 0.0
        return self.successes / self.invocations

    @property
    def feedback_score(self) -> int:
        """``feedback_good - feedback_bad``."""
        return self.feedback_good - self.feedback_bad

    @property
    def p50_ms(self) -> float:
        """Median latency in milliseconds (0.0 when no data)."""
        return self._percentile(0.5)

    @property
    def p95_ms(self) -> float:
        """95th-percentile latency in milliseconds (0.0 when no data)."""
        return self._percentile(0.95)

    def _percentile(self, q: float) -> float:
        """Return the *q*-quantile of the sorted ``durations_ms`` buffer.

        Args:
            q: Quantile in ``[0.0, 1.0]`` (e.g. ``0.5`` for the median).

        Returns:
            The sampled duration in milliseconds, or ``0.0`` when no
            measurements have been recorded.
        """
        if not self.durations_ms:
            return 0.0
        idx = max(0, min(len(self.durations_ms) - 1, round(q * (len(self.durations_ms) - 1))))
        return float(self.durations_ms[idx])


class SkillTelemetry:
    """Thread-safe aggregator for skill-related audit events.

    Wire it into the audit collector via :meth:`record`. Inspect via
    :meth:`stats` (single skill) or :meth:`all_stats` (all known).

    Example:
        >>> tel = SkillTelemetry()
        >>> tel.record(SkillUsedEvent(skill_name="a", outcome="success", duration_ms=12))
        >>> assert tel.stats("a").invocations == 1
    """

    def __init__(self) -> None:
        """Initialise an empty telemetry registry."""
        self._stats: dict[str, SkillStats] = {}
        self._lock = threading.Lock()

    def record(self, event: tp.Any) -> None:
        """Update stats for any of the three skill-related event types.

        Other event types are ignored, so this can safely be wired into
        a generic audit collector that fans out to multiple consumers.
        """
        with self._lock:
            if isinstance(event, SkillUsedEvent):
                self._on_used(event)
            elif isinstance(event, SkillFeedbackEvent):
                self._on_feedback(event)
            elif isinstance(event, SkillAuthoredEvent):
                self._on_authored(event)

    def _entry(self, name: str) -> SkillStats:
        """Return the :class:`SkillStats` for *name*, creating it on first use.

        Args:
            name: Canonical skill name.

        Returns:
            The mutable stats record stored for this skill.
        """
        s = self._stats.get(name)
        if s is None:
            s = SkillStats(skill_name=name)
            self._stats[name] = s
        return s

    def _on_used(self, ev: SkillUsedEvent) -> None:
        """Fold a :class:`SkillUsedEvent` into its stats entry.

        Increments invocation counters, captures the outcome, updates the
        latest version, and inserts the duration into the sorted
        percentile buffer.

        Args:
            ev: The usage event to apply.
        """
        s = self._entry(ev.skill_name)
        s.invocations += 1
        s.last_invoked = datetime.now()
        if ev.version:
            s.version = ev.version
        if ev.outcome == "success":
            s.successes += 1
        else:
            s.failures += 1
            if ev.outcome:
                s.last_failure_reason = ev.outcome
        if ev.duration_ms > 0:
            bisect.insort(s.durations_ms, float(ev.duration_ms))

    def _on_feedback(self, ev: SkillFeedbackEvent) -> None:
        """Fold a :class:`SkillFeedbackEvent` into the good/bad tally.

        Args:
            ev: Feedback event carrying a ``"good"`` or ``"bad"`` rating.
        """
        s = self._entry(ev.skill_name)
        if ev.rating == "good":
            s.feedback_good += 1
        elif ev.rating == "bad":
            s.feedback_bad += 1

    def _on_authored(self, ev: SkillAuthoredEvent) -> None:
        """Record a new authoring event, updating version and authored time.

        Args:
            ev: The authoring event emitted by the pipeline.
        """
        s = self._entry(ev.skill_name)
        s.version = ev.version or s.version
        s.authored_at = datetime.now()

    def stats(self, skill_name: str) -> SkillStats | None:
        """Return the stats entry for *skill_name* or ``None`` if unknown."""
        with self._lock:
            return self._stats.get(skill_name)

    def all_stats(self) -> dict[str, SkillStats]:
        """Return a snapshot mapping skill name → stats."""
        with self._lock:
            return dict(self._stats)

    def candidates_for_deprecation(
        self,
        *,
        min_invocations: int = 5,
        max_success_rate: float = 0.4,
    ) -> list[str]:
        """Return skill names that look like good auto-deprecation candidates.

        Heuristic: at least ``min_invocations`` recorded uses with a
        success rate at or below ``max_success_rate``.

        Args:
            min_invocations: Required minimum invocation count to avoid
                evaluating skills with insufficient data.
            max_success_rate: Skills at or below this rate are flagged.

        Returns:
            List of skill names sorted by ascending success rate.
        """
        with self._lock:
            flagged = [
                (s.success_rate, s.skill_name)
                for s in self._stats.values()
                if s.invocations >= min_invocations and s.success_rate <= max_success_rate
            ]
        flagged.sort()
        return [name for _, name in flagged]
