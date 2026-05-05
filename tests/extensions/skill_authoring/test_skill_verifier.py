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
"""Tests for SkillVerifier."""

from __future__ import annotations

from xerxes.extensions.skill_authoring import (
    SkillVerifier,
    ToolSequenceTracker,
    VerificationStep,
)


def _candidate(prompt="ci setup"):
    t = ToolSequenceTracker()
    t.begin_turn(user_prompt=prompt)
    t.record_call("Read", {"path": "ci.yml"})
    t.record_call("Edit", {"path": "ci.yml", "old": "x", "new": "y"})
    t.record_call("Bash", {"cmd": "pytest"})
    return t.end_turn()


class TestSkillVerifier:
    def test_generate_includes_call_count(self):
        v = SkillVerifier()
        steps = v.generate(_candidate())
        kinds = [s.kind for s in steps]
        assert "call_count" in kinds

    def test_generate_includes_position_for_each_call(self):
        v = SkillVerifier()
        steps = v.generate(_candidate())
        positions = [s for s in steps if s.kind == "sequence_position"]
        assert len(positions) == 3
        assert [s.tool_name for s in positions] == ["Read", "Edit", "Bash"]

    def test_verify_passes_on_identical_sequence(self):
        v = SkillVerifier()
        c = _candidate()
        steps = v.generate(c)
        result = v.verify(steps, c)
        assert result.passed is True
        assert result.failed_steps == []

    def test_verify_fails_on_wrong_count(self):
        v = SkillVerifier()
        steps = v.generate(_candidate())
        t = ToolSequenceTracker()
        t.begin_turn()
        t.record_call("Read", {"path": "ci.yml"})
        t.record_call("Edit", {"path": "ci.yml", "old": "x", "new": "y"})
        result = v.verify(steps, t.end_turn())
        assert result.passed is False
        assert any("expected 3" in r for _, r in result.failed_steps)

    def test_verify_fails_on_wrong_tool_at_position(self):
        v = SkillVerifier()
        steps = v.generate(_candidate())
        t = ToolSequenceTracker()
        t.begin_turn()
        t.record_call("Read", {"path": "ci.yml"})
        t.record_call("Bash", {"cmd": "ls"})
        t.record_call("Bash", {"cmd": "pytest"})
        result = v.verify(steps, t.end_turn())
        assert result.passed is False

    def test_verify_args_subset_passes_on_superset(self):
        v = SkillVerifier()
        c = _candidate()
        steps = v.generate(c)
        t = ToolSequenceTracker()
        t.begin_turn()
        t.record_call("Read", {"path": "ci.yml", "extra": "ok"})
        t.record_call("Edit", {"path": "ci.yml", "old": "x", "new": "y", "extra": "ok"})
        t.record_call("Bash", {"cmd": "pytest", "extra": "ok"})
        result = v.verify(steps, t.end_turn())
        assert result.passed is True

    def test_verify_args_subset_fails_on_missing_key(self):
        v = SkillVerifier()
        steps = [VerificationStep(kind="args_subset", tool_name="Edit", position=0, args_required={"path": "x"})]
        t = ToolSequenceTracker()
        t.begin_turn()
        t.record_call("Edit", {"old": "y"})
        result = v.verify(steps, t.end_turn())
        assert result.passed is False

    def test_unknown_kind_fails(self):
        v = SkillVerifier()
        result = v.verify([VerificationStep(kind="not_a_thing")], _candidate())
        assert result.passed is False
