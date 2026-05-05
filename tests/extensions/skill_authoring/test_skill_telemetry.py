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
"""Tests for SkillTelemetry and skill audit events."""

from __future__ import annotations

from xerxes.audit import (
    AuditEmitter,
    InMemoryCollector,
    SkillAuthoredEvent,
    SkillFeedbackEvent,
    SkillUsedEvent,
)
from xerxes.extensions.skill_authoring import SkillTelemetry


class TestSkillEventEmitters:
    def test_emit_skill_used(self):
        col = InMemoryCollector()
        em = AuditEmitter(collector=col)
        em.emit_skill_used(skill_name="ci-setup", outcome="success", duration_ms=42.0)
        assert any(isinstance(e, SkillUsedEvent) for e in col.get_events())
        ev = next(e for e in col.get_events() if isinstance(e, SkillUsedEvent))
        assert ev.skill_name == "ci-setup"
        assert ev.outcome == "success"
        assert ev.duration_ms == 42.0

    def test_emit_skill_authored(self):
        col = InMemoryCollector()
        em = AuditEmitter(collector=col)
        em.emit_skill_authored(
            skill_name="x",
            version="0.1.0",
            source_path="/skills/x/SKILL.md",
            tool_count=5,
            unique_tools=["A", "B"],
        )
        ev = next(e for e in col.get_events() if isinstance(e, SkillAuthoredEvent))
        assert ev.tool_count == 5

    def test_emit_skill_feedback(self):
        col = InMemoryCollector()
        em = AuditEmitter(collector=col)
        em.emit_skill_feedback(skill_name="x", rating="bad", reason="produced wrong output")
        ev = next(e for e in col.get_events() if isinstance(e, SkillFeedbackEvent))
        assert ev.rating == "bad"
        assert ev.reason == "produced wrong output"


class TestSkillTelemetry:
    def test_records_used_event(self):
        tel = SkillTelemetry()
        tel.record(SkillUsedEvent(skill_name="a", outcome="success", duration_ms=10))
        s = tel.stats("a")
        assert s.invocations == 1
        assert s.successes == 1
        assert s.success_rate == 1.0

    def test_failure_recorded(self):
        tel = SkillTelemetry()
        tel.record(SkillUsedEvent(skill_name="a", outcome="failure", duration_ms=5))
        s = tel.stats("a")
        assert s.failures == 1
        assert s.success_rate == 0.0
        assert s.last_failure_reason == "failure"

    def test_p50_p95(self):
        tel = SkillTelemetry()
        for d in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            tel.record(SkillUsedEvent(skill_name="a", outcome="success", duration_ms=d))
        s = tel.stats("a")
        assert 40 <= s.p50_ms <= 60
        assert s.p95_ms >= 90

    def test_feedback_score(self):
        tel = SkillTelemetry()
        tel.record(SkillFeedbackEvent(skill_name="a", rating="good"))
        tel.record(SkillFeedbackEvent(skill_name="a", rating="good"))
        tel.record(SkillFeedbackEvent(skill_name="a", rating="bad"))
        assert tel.stats("a").feedback_score == 1

    def test_authored_event_sets_authored_at(self):
        tel = SkillTelemetry()
        tel.record(SkillAuthoredEvent(skill_name="a", version="0.1.0"))
        s = tel.stats("a")
        assert s.authored_at is not None
        assert s.version == "0.1.0"

    def test_unrelated_event_ignored(self):
        from xerxes.audit.events import ErrorEvent

        tel = SkillTelemetry()
        tel.record(ErrorEvent(error_type="X"))
        assert tel.all_stats() == {}

    def test_candidates_for_deprecation(self):
        tel = SkillTelemetry()
        for outcome in ["failure"] * 8 + ["success"] * 2:
            tel.record(SkillUsedEvent(skill_name="bad", outcome=outcome))
        for outcome in ["success"] * 9 + ["failure"] * 1:
            tel.record(SkillUsedEvent(skill_name="good", outcome=outcome))
        flagged = tel.candidates_for_deprecation(min_invocations=5, max_success_rate=0.5)
        assert "bad" in flagged
        assert "good" not in flagged

    def test_unknown_skill_returns_none(self):
        tel = SkillTelemetry()
        assert tel.stats("nope") is None

    def test_thread_safe_concurrent_record(self):
        import threading

        tel = SkillTelemetry()

        def hammer():
            for _ in range(100):
                tel.record(SkillUsedEvent(skill_name="x", outcome="success", duration_ms=1))

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert tel.stats("x").invocations == 400
