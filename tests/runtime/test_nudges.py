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
"""Tests for xerxes.runtime.nudges."""

from __future__ import annotations

from xerxes.runtime.nudges import (
    MemoryNudge,
    NudgeContext,
    NudgeManager,
    NudgeRule,
    SkillNudge,
)


class TestMemoryNudge:
    def test_fires_on_interval_with_durable_hint(self):
        n = MemoryNudge(interval=4)
        ctx = NudgeContext(turn_index=3, last_user_message="please remember my deadline")
        assert n.should_fire(ctx) is True

    def test_doesnt_fire_without_durable_hint(self):
        n = MemoryNudge(interval=4)
        ctx = NudgeContext(turn_index=3, last_user_message="what's the time?")
        assert n.should_fire(ctx) is False

    def test_doesnt_fire_off_interval(self):
        n = MemoryNudge(interval=4)
        ctx = NudgeContext(turn_index=2, last_user_message="remember this")
        assert n.should_fire(ctx) is False

    def test_doesnt_fire_when_memory_writes_happened(self):
        n = MemoryNudge(interval=4)
        ctx = NudgeContext(
            turn_index=3,
            last_user_message="remember this",
            memory_writes_since_last_fire=1,
        )
        assert n.should_fire(ctx) is False

    def test_message_mentions_save_memory(self):
        n = MemoryNudge()
        msg = n.message(NudgeContext(turn_index=0))
        assert "save_memory" in msg


class TestSkillNudge:
    def test_fires_above_threshold(self):
        n = SkillNudge(threshold=6)
        ctx = NudgeContext(turn_index=0, successful_tool_calls_this_turn=6)
        assert n.should_fire(ctx) is True

    def test_doesnt_fire_below_threshold(self):
        n = SkillNudge(threshold=6)
        ctx = NudgeContext(turn_index=0, successful_tool_calls_this_turn=5)
        assert n.should_fire(ctx) is False


class TestNudgeManager:
    def test_default_rules(self):
        m = NudgeManager()
        names = {r.name for r in m.rules}
        assert names == {"memory", "skill"}

    def test_check_collects_fires(self):
        m = NudgeManager([SkillNudge(threshold=3)])
        out = m.check(NudgeContext(turn_index=0, successful_tool_calls_this_turn=5))
        assert len(out) == 1
        assert out[0][0] == "skill"

    def test_disabled_rules_skipped(self):
        m = NudgeManager()
        m.disable("memory")
        ctx = NudgeContext(turn_index=3, last_user_message="please remember")
        out = m.check(ctx)
        # Memory rule was the only one that would have fired.
        assert all(name != "memory" for name, _ in out)

    def test_re_enable(self):
        m = NudgeManager()
        m.disable("memory")
        m.enable("memory")
        assert "memory" not in m.disabled()

    def test_fired_count_increments(self):
        m = NudgeManager([SkillNudge(threshold=1)])
        for _ in range(3):
            m.check(NudgeContext(turn_index=0, successful_tool_calls_this_turn=1))
        assert m.fired_count("skill") == 3

    def test_custom_rule(self):
        class _Always(NudgeRule):
            name = "always"

            def should_fire(self, ctx):
                return True

            def message(self, ctx):
                return "always"

        m = NudgeManager([_Always()])
        out = m.check(NudgeContext(turn_index=0))
        assert out == [("always", "always")]
