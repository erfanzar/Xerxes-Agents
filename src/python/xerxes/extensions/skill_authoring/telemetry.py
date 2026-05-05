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
        skill_name (str): IN: Skill identifier. OUT: Stored.
        version (str): IN: Current version. OUT: Updated on use / author.
        invocations (int): IN: Use counter. OUT: Incremented by ``record``.
        successes (int): IN: Success counter. OUT: Incremented by ``record``.
        failures (int): IN: Failure counter. OUT: Incremented by ``record``.
        durations_ms (list[float]): IN: Empty initially. OUT: Kept sorted for
            percentile queries.
        last_invoked (datetime | None): IN: ``None`` initially. OUT: Updated
            on use.
        last_failure_reason (str): IN: Empty initially. OUT: Updated on
            failure.
        feedback_good (int): IN: Zero initially. OUT: Incremented on positive
            feedback.
        feedback_bad (int): IN: Zero initially. OUT: Incremented on negative
            feedback.
        authored_at (datetime | None): IN: ``None`` initially. OUT: Set on
            author event.
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
        """Return the ratio of successes to total invocations.

        Returns:
            float: OUT: 0.0 if no invocations, else ``successes / invocations``.
        """

        if self.invocations == 0:
            return 0.0
        return self.successes / self.invocations

    @property
    def feedback_score(self) -> int:
        """Return the net feedback score.

        Returns:
            int: OUT: ``feedback_good - feedback_bad``.
        """

        return self.feedback_good - self.feedback_bad

    @property
    def p50_ms(self) -> float:
        """Return the 50th percentile duration in milliseconds.

        Returns:
            float: OUT: Median duration or 0.0 if no data.
        """

        return self._percentile(0.5)

    @property
    def p95_ms(self) -> float:
        """Return the 95th percentile duration in milliseconds.

        Returns:
            float: OUT: 95th percentile duration or 0.0 if no data.
        """

        return self._percentile(0.95)

    def _percentile(self, q: float) -> float:
        """Compute a percentile from the sorted duration list.

        Args:
            q (float): IN: Quantile in [0.0, 1.0]. OUT: Used to index the
                sorted durations.

        Returns:
            float: OUT: Duration at the requested percentile.
        """

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
        """Process an audit event and update internal metrics.

        Args:
            event (tp.Any): IN: Expected to be a ``SkillUsedEvent``,
                ``SkillFeedbackEvent``, or ``SkillAuthoredEvent``. OUT:
                Dispatched to the appropriate handler.

        Returns:
            None: OUT: Internal stats are updated under ``_lock``.
        """

        with self._lock:
            if isinstance(event, SkillUsedEvent):
                self._on_used(event)
            elif isinstance(event, SkillFeedbackEvent):
                self._on_feedback(event)
            elif isinstance(event, SkillAuthoredEvent):
                self._on_authored(event)

    def _entry(self, name: str) -> SkillStats:
        """Get or create a ``SkillStats`` record by name.

        Args:
            name (str): IN: Skill identifier. OUT: Lookup key.

        Returns:
            SkillStats: OUT: Existing or newly created stats object.
        """

        s = self._stats.get(name)
        if s is None:
            s = SkillStats(skill_name=name)
            self._stats[name] = s
        return s

    def _on_used(self, ev: SkillUsedEvent) -> None:
        """Update stats from a ``SkillUsedEvent``.

        Args:
            ev (SkillUsedEvent): IN: Usage event. OUT: Fields are copied into
                the stats record.

        Returns:
            None: OUT: Stats are mutated.
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
        """Update stats from a ``SkillFeedbackEvent``.

        Args:
            ev (SkillFeedbackEvent): IN: Feedback event. OUT: Rating is
                applied to the stats record.

        Returns:
            None: OUT: Stats are mutated.
        """

        s = self._entry(ev.skill_name)
        if ev.rating == "good":
            s.feedback_good += 1
        elif ev.rating == "bad":
            s.feedback_bad += 1

    def _on_authored(self, ev: SkillAuthoredEvent) -> None:
        """Update stats from a ``SkillAuthoredEvent``.

        Args:
            ev (SkillAuthoredEvent): IN: Authorship event. OUT: Version and
                timestamp are updated.

        Returns:
            None: OUT: Stats are mutated.
        """

        s = self._entry(ev.skill_name)
        s.version = ev.version or s.version
        s.authored_at = datetime.now()

    def stats(self, skill_name: str) -> SkillStats | None:
        """Retrieve stats for a single skill.

        Args:
            skill_name (str): IN: Skill identifier. OUT: Looked up under
                ``_lock``.

        Returns:
            SkillStats | None: OUT: Stats snapshot or ``None``.
        """

        with self._lock:
            return self._stats.get(skill_name)

    def all_stats(self) -> dict[str, SkillStats]:
        """Return a snapshot of all skill stats.

        Returns:
            dict[str, SkillStats]: OUT: Copy of the internal mapping.
        """

        with self._lock:
            return dict(self._stats)

    def candidates_for_deprecation(
        self,
        *,
        min_invocations: int = 5,
        max_success_rate: float = 0.4,
    ) -> list[str]:
        """List skill names that fall below the success-rate threshold.

        Args:
            min_invocations (int): IN: Minimum sample size. OUT: Filters out
                rarely-used skills.
            max_success_rate (float): IN: Success rate ceiling. OUT: Skills at
                or below this rate are flagged.

        Returns:
            list[str]: OUT: Flagged skill names sorted by success rate
            ascending.
        """

        with self._lock:
            flagged = [
                (s.success_rate, s.skill_name)
                for s in self._stats.values()
                if s.invocations >= min_invocations and s.success_rate <= max_success_rate
            ]
        flagged.sort()
        return [name for _, name in flagged]
