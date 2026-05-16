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
"""Skill usage telemetry and statistics aggregation.

``SkillTelemetry`` consumes audit events and maintains per-skill success rates,
duration percentiles, and feedback scores.
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
    """Aggregated metrics for a single skill.

    Attributes:
        skill_name: Skill identifier.
        version: Current version, updated on use and authorship.
        invocations: Total number of use events recorded.
        successes: Number of successful invocations.
        failures: Number of failed invocations.
        durations_ms: Sorted list of observed durations, enabling percentile queries.
        last_invoked: Timestamp of the most recent use event.
        last_failure_reason: Outcome message from the most recent failure.
        feedback_good: Count of positive feedback events.
        feedback_bad: Count of negative feedback events.
        authored_at: Timestamp of the last authorship event.
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
        """Return ``successes / invocations``, or ``0.0`` if never invoked."""

        if self.invocations == 0:
            return 0.0
        return self.successes / self.invocations

    @property
    def feedback_score(self) -> int:
        """Return ``feedback_good - feedback_bad``."""

        return self.feedback_good - self.feedback_bad

    @property
    def p50_ms(self) -> float:
        """Return the 50th percentile of observed durations in ms."""

        return self._percentile(0.5)

    @property
    def p95_ms(self) -> float:
        """Return the 95th percentile of observed durations in ms."""

        return self._percentile(0.95)

    def _percentile(self, q: float) -> float:
        """Return the duration at quantile ``q`` (0.0 - 1.0) from the sorted samples."""

        if not self.durations_ms:
            return 0.0
        idx = max(0, min(len(self.durations_ms) - 1, round(q * (len(self.durations_ms) - 1))))
        return float(self.durations_ms[idx])


class SkillTelemetry:
    """Thread-safe aggregator of skill-related audit events."""

    def __init__(self) -> None:
        """Initialize empty stats and a threading lock."""

        self._stats: dict[str, SkillStats] = {}
        self._lock = threading.Lock()

    def record(self, event: tp.Any) -> None:
        """Dispatch ``event`` to the appropriate stats handler under the lock.

        Recognises ``SkillUsedEvent``, ``SkillFeedbackEvent``, and
        ``SkillAuthoredEvent``; unrecognised events are ignored.
        """

        with self._lock:
            if isinstance(event, SkillUsedEvent):
                self._on_used(event)
            elif isinstance(event, SkillFeedbackEvent):
                self._on_feedback(event)
            elif isinstance(event, SkillAuthoredEvent):
                self._on_authored(event)

    def _entry(self, name: str) -> SkillStats:
        """Return the stats record for ``name``, creating one if needed."""

        s = self._stats.get(name)
        if s is None:
            s = SkillStats(skill_name=name)
            self._stats[name] = s
        return s

    def _on_used(self, ev: SkillUsedEvent) -> None:
        """Apply a ``SkillUsedEvent`` to the relevant stats record."""

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
        """Increment the appropriate feedback counter from ``ev``."""

        s = self._entry(ev.skill_name)
        if ev.rating == "good":
            s.feedback_good += 1
        elif ev.rating == "bad":
            s.feedback_bad += 1

    def _on_authored(self, ev: SkillAuthoredEvent) -> None:
        """Refresh the version and authorship timestamp from ``ev``."""

        s = self._entry(ev.skill_name)
        s.version = ev.version or s.version
        s.authored_at = datetime.now()

    def stats(self, skill_name: str) -> SkillStats | None:
        """Return the stats record for ``skill_name``, or ``None``."""

        with self._lock:
            return self._stats.get(skill_name)

    def all_stats(self) -> dict[str, SkillStats]:
        """Return a copy of the internal stats mapping."""

        with self._lock:
            return dict(self._stats)

    def candidates_for_deprecation(
        self,
        *,
        min_invocations: int = 5,
        max_success_rate: float = 0.4,
    ) -> list[str]:
        """Return skill names whose success rate falls at or below the threshold.

        Args:
            min_invocations: Minimum sample size for a skill to be considered.
            max_success_rate: Skills at or below this success rate are flagged.

        Returns:
            Flagged skill names sorted by success rate, ascending.
        """

        with self._lock:
            flagged = [
                (s.success_rate, s.skill_name)
                for s in self._stats.values()
                if s.invocations >= min_invocations and s.success_rate <= max_success_rate
            ]
        flagged.sort()
        return [name for _, name in flagged]
