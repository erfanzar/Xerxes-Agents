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
"""Tests for ToolSequenceTracker and SkillAuthoringTrigger."""

from __future__ import annotations

import pytest
from xerxes.extensions.skill_authoring import (
    SkillAuthoringConfig,
    SkillAuthoringTrigger,
    ToolSequenceTracker,
)


@pytest.fixture
def tracker():
    t = ToolSequenceTracker()
    t.begin_turn(agent_id="coder", turn_id="t1", user_prompt="set up CI")
    return t


class TestToolSequenceTracker:
    def test_records_basic_call(self, tracker):
        tracker.record_call("Read", {"path": "ci.yml"})
        assert tracker.call_count == 1
        assert tracker.events[0].tool_name == "Read"

    def test_unique_tools(self, tracker):
        tracker.record_call("Read", {"p": 1})
        tracker.record_call("Write", {"p": 2})
        tracker.record_call("Read", {"p": 3})
        candidate = tracker.end_turn(final_response="done")
        assert candidate.unique_tools == ["Read", "Write"]

    def test_detects_retry(self, tracker):
        tracker.record_call("Bash", {"cmd": "ls"}, status="error")
        tracker.record_call("Bash", {"cmd": "ls"}, status="success")
        candidate = tracker.end_turn()
        assert candidate.retries == 1
        assert candidate.events[1].retry_of == 0

    def test_signature(self, tracker):
        tracker.record_call("A", {})
        tracker.record_call("B", {})
        tracker.record_call("C", {})
        c = tracker.end_turn()
        assert c.signature() == "A>B>C"

    def test_total_duration(self, tracker):
        tracker.record_call("A", {}, duration_ms=100)
        tracker.record_call("B", {}, duration_ms=250)
        c = tracker.end_turn()
        assert c.total_duration_ms == 350

    def test_begin_turn_resets(self, tracker):
        tracker.record_call("X", {})
        tracker.begin_turn(agent_id="other", user_prompt="new")
        assert tracker.call_count == 0
        assert tracker._agent_id == "other"

    def test_mark_call_start_measures_duration(self, tracker):
        import time

        tracker.mark_call_start()
        time.sleep(0.005)
        ev = tracker.record_call("Slow", {})
        assert ev.duration_ms >= 4.0

    def test_end_turn_returns_candidate_with_metadata(self, tracker):
        tracker.record_call("X", {})
        c = tracker.end_turn(final_response="done")
        assert c.agent_id == "coder"
        assert c.turn_id == "t1"
        assert c.user_prompt == "set up CI"
        assert c.final_response == "done"


class TestSkillAuthoringTrigger:
    def _candidate(self, n_calls=5, n_unique=3, n_retries=0, fail_terminal=False):
        t = ToolSequenceTracker()
        t.begin_turn()
        for i in range(n_calls):
            tool = f"T{i % n_unique}"
            t.record_call(tool, {"i": i})
        for _ in range(n_retries):
            t.record_call("Tretry", {"x": 1}, status="error")
            t.record_call("Tretry", {"x": 1}, status="success")
        if fail_terminal:
            t.record_call("Bad", {}, status="error")
        return t.end_turn()

    def test_default_config_accepts_5_call_3_unique(self):
        trig = SkillAuthoringTrigger()
        c = self._candidate(n_calls=5, n_unique=3)
        assert trig.should_author(c) is True

    def test_too_few_calls_rejected(self):
        trig = SkillAuthoringTrigger()
        c = self._candidate(n_calls=3, n_unique=3)
        assert trig.should_author(c) is False
        assert "tool calls" in trig.reason(c)

    def test_too_few_unique_tools_rejected(self):
        trig = SkillAuthoringTrigger()
        c = self._candidate(n_calls=5, n_unique=1)
        assert trig.should_author(c) is False

    def test_terminal_failure_rejected(self):
        trig = SkillAuthoringTrigger()
        c = self._candidate(n_calls=5, n_unique=3, fail_terminal=True)
        assert trig.should_author(c) is False

    def test_recovered_failures_allowed(self):
        trig = SkillAuthoringTrigger(SkillAuthoringConfig(max_retries_allowed=5))
        c = self._candidate(n_calls=5, n_unique=3, n_retries=1)
        assert trig.should_author(c) is True

    def test_too_many_retries_rejected(self):
        trig = SkillAuthoringTrigger(SkillAuthoringConfig(max_retries_allowed=1))
        c = self._candidate(n_calls=5, n_unique=3, n_retries=3)
        assert trig.should_author(c) is False

    def test_disabled_returns_false(self):
        trig = SkillAuthoringTrigger(SkillAuthoringConfig(enabled=False))
        c = self._candidate(n_calls=10, n_unique=4)
        assert trig.should_author(c) is False
        assert "disabled" in trig.reason(c)

    def test_existing_skill_skips_when_configured(self, tmp_path):
        from xerxes.extensions.skills import Skill, SkillMetadata, SkillRegistry

        registry = SkillRegistry()
        registry._skills["covers"] = Skill(
            metadata=SkillMetadata(name="covers", tags=["T0", "T1", "T2"], required_tools=[]),
            instructions="...",
            source_path=tmp_path / "SKILL.md",
        )
        trig = SkillAuthoringTrigger(skill_registry=registry)
        c = self._candidate(n_calls=5, n_unique=3)
        assert trig.should_author(c) is False
