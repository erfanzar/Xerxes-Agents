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
"""Tests for SkillAuthoringPipeline + SkillImprover end-to-end."""

from __future__ import annotations

import json

import pytest
from xerxes.audit import AuditEmitter, InMemoryCollector, SkillAuthoredEvent
from xerxes.extensions.skill_authoring import (
    SkillAuthoringConfig,
    SkillAuthoringPipeline,
    SkillImprover,
    ToolSequenceTracker,
)


@pytest.fixture
def pipeline(tmp_path):
    em = AuditEmitter(collector=InMemoryCollector())
    return SkillAuthoringPipeline(
        skills_dir=tmp_path / "skills",
        config=SkillAuthoringConfig(min_tool_calls=3, min_unique_tools=2),
        audit_emitter=em,
    )


class TestPipeline:
    def test_records_and_authors_skill(self, pipeline, tmp_path):
        pipeline.begin_turn(agent_id="coder", user_prompt="set up CI")
        pipeline.record_call("Read", {"path": "ci.yml"})
        pipeline.record_call("Edit", {"path": "ci.yml"})
        pipeline.record_call("Bash", {"cmd": "pytest"})
        result = pipeline.on_turn_end(final_response="done")
        assert result.authored is True
        assert result.skill_path is not None
        assert result.skill_path.exists()
        assert result.recipe_path is not None
        assert result.recipe_path.exists()

    def test_recipe_is_valid_json(self, pipeline):
        pipeline.begin_turn(user_prompt="x")
        pipeline.record_call("A", {})
        pipeline.record_call("B", {})
        pipeline.record_call("C", {})
        result = pipeline.on_turn_end()
        assert result.authored
        steps = json.loads(result.recipe_path.read_text())
        assert isinstance(steps, list)
        assert any(s.get("kind") == "call_count" for s in steps)

    def test_skips_when_below_threshold(self, pipeline):
        pipeline.begin_turn(user_prompt="trivial")
        pipeline.record_call("A", {})
        result = pipeline.on_turn_end()
        assert result.authored is False
        assert "tool calls" in result.reason

    def test_audit_event_emitted(self, pipeline):
        pipeline.begin_turn(user_prompt="set up CI")
        pipeline.record_call("Read", {})
        pipeline.record_call("Edit", {})
        pipeline.record_call("Bash", {})
        pipeline.on_turn_end()
        events = pipeline.audit_emitter.collector.get_events()
        assert any(isinstance(e, SkillAuthoredEvent) for e in events)


class TestSkillImprover:
    def test_bumps_patch_and_writes_backup(self, tmp_path):
        skill = tmp_path / "demo" / "SKILL.md"
        skill.parent.mkdir()
        skill.write_text(
            "---\nname: demo\nversion: 0.1.0\n---\n# Old body\n",
            encoding="utf-8",
        )
        t = ToolSequenceTracker()
        t.begin_turn(user_prompt="updated procedure")
        t.record_call("Read", {})
        t.record_call("Write", {})
        candidate = t.end_turn()

        imp = SkillImprover()
        result = imp.improve(skill, candidate)
        assert result.improved is True
        assert result.old_version == "0.1.0"
        assert result.new_version == "0.1.1"
        assert result.backup_path is not None
        assert result.backup_path.exists()
        new_text = skill.read_text()
        assert "version: 0.1.1" in new_text
        assert "# Procedure" in new_text

    def test_missing_path_returns_failure(self, tmp_path):
        t = ToolSequenceTracker()
        t.begin_turn()
        t.record_call("X", {})
        c = t.end_turn()
        result = SkillImprover().improve(tmp_path / "missing" / "SKILL.md", c)
        assert result.improved is False
        assert "missing" in result.reason

    def test_unparseable_version_falls_back(self, tmp_path):
        skill = tmp_path / "x" / "SKILL.md"
        skill.parent.mkdir()
        skill.write_text("---\nname: x\nversion: not-a-version\n---\nbody\n")
        t = ToolSequenceTracker()
        t.begin_turn()
        t.record_call("Y", {})
        result = SkillImprover().improve(skill, t.end_turn())
        assert result.improved is True
        assert result.new_version == "0.1.1"


class TestStreamingEvent:
    def test_skill_suggestion_in_stream_event_union(self):
        from xerxes.streaming.events import SkillSuggestion, StreamEvent

        ev = SkillSuggestion(skill_name="x", description="d", source_path="/tmp/x")
        assert isinstance(ev, StreamEvent.__args__)  # type: ignore[attr-defined]
