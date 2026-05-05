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
"""Tests for SkillVariantPicker and SkillLifecycleManager."""

from __future__ import annotations

from xerxes.audit.events import SkillUsedEvent
from xerxes.extensions.skill_authoring import (
    SkillLifecycleManager,
    SkillTelemetry,
    SkillVariant,
    SkillVariantPicker,
)
from xerxes.extensions.skills import SkillRegistry


class TestSkillVariantPicker:
    def test_no_variant_returns_base(self):
        p = SkillVariantPicker()
        assert p.pick("base") == "base"

    def test_full_rollout_routes_all(self):
        p = SkillVariantPicker()
        p.add(SkillVariant("base", "v2", rollout=1.0))
        for u in ("alice", "bob", "carol"):
            assert p.pick("base", user_id=u) == "v2"

    def test_zero_rollout_keeps_base(self):
        p = SkillVariantPicker()
        p.add(SkillVariant("base", "v2", rollout=0.0))
        for u in ("alice", "bob"):
            assert p.pick("base", user_id=u) == "base"

    def test_split_is_deterministic_per_user(self):
        p = SkillVariantPicker()
        p.add(SkillVariant("base", "v2", rollout=0.5))
        first = p.pick("base", user_id="alice")
        for _ in range(10):
            assert p.pick("base", user_id="alice") == first

    def test_split_distributes_users(self):
        p = SkillVariantPicker()
        p.add(SkillVariant("base", "v2", rollout=0.5))
        seen_base = sum(1 for i in range(200) if p.pick("base", user_id=f"u{i}") == "base")
        assert 60 <= seen_base <= 140  # approx 50% with reasonable variance

    def test_rollout_clamped_to_unit(self):
        p = SkillVariantPicker()
        p.add(SkillVariant("base", "v2", rollout=2.0))
        assert p.pick("base", user_id="x") == "v2"

    def test_remove(self):
        p = SkillVariantPicker()
        p.add(SkillVariant("base", "v2", rollout=1.0))
        p.remove("base")
        assert p.pick("base", user_id="x") == "base"


class TestLifecycleManager:
    def _flood(self, tel, name, *, success, total):
        for _ in range(success):
            tel.record(SkillUsedEvent(skill_name=name, outcome="success"))
        for _ in range(total - success):
            tel.record(SkillUsedEvent(skill_name=name, outcome="failure"))

    def test_evaluate_returns_proposed(self):
        tel = SkillTelemetry()
        self._flood(tel, "bad", success=2, total=10)
        mgr = SkillLifecycleManager(tel, min_invocations=5, max_success_rate=0.5)
        decisions = mgr.evaluate()
        assert any(d.skill_name == "bad" for d in decisions)
        assert all(d.action == "proposed" for d in decisions)

    def test_apply_renames_skill_md(self, tmp_path):
        skill_dir = tmp_path / "ci-bad"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: ci-bad\nversion: 0.1.0\n---\nbody",
            encoding="utf-8",
        )
        tel = SkillTelemetry()
        self._flood(tel, "ci-bad", success=2, total=10)
        mgr = SkillLifecycleManager(
            tel,
            registry=None,
            skills_dir=tmp_path,
            min_invocations=5,
            max_success_rate=0.5,
        )
        applied = mgr.apply()
        deprecated = [d for d in applied if d.action == "deprecated"]
        assert len(deprecated) == 1
        assert not (skill_dir / "SKILL.md").exists()
        assert (skill_dir / "SKILL.deprecated.md").exists()

    def test_apply_skips_missing(self, tmp_path):
        tel = SkillTelemetry()
        self._flood(tel, "ghost", success=0, total=10)
        mgr = SkillLifecycleManager(tel, registry=None, skills_dir=tmp_path, min_invocations=5, max_success_rate=0.5)
        applied = mgr.apply()
        assert any(d.action == "missing" for d in applied)

    def test_registry_discovery_skips_deprecated(self, tmp_path):
        good = tmp_path / "good"
        good.mkdir()
        (good / "SKILL.md").write_text("---\nname: good\nversion: 0.1.0\n---\nbody")
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "SKILL.deprecated.md").write_text("---\nname: bad\nversion: 0.1.0\n---\nbody")
        reg = SkillRegistry()
        reg.discover(tmp_path)
        names = {s.name for s in reg.get_all()}
        assert "good" in names
        assert "bad" not in names

    def test_high_success_rate_not_deprecated(self, tmp_path):
        skill_dir = tmp_path / "good"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: good\nversion: 0.1.0\n---\nbody")
        tel = SkillTelemetry()
        self._flood(tel, "good", success=9, total=10)
        mgr = SkillLifecycleManager(tel, registry=None, skills_dir=tmp_path, min_invocations=5, max_success_rate=0.5)
        applied = mgr.apply()
        assert applied == []
        assert (skill_dir / "SKILL.md").exists()
