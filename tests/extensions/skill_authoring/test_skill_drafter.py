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
"""Tests for SkillDrafter and template renderer."""

from __future__ import annotations

from xerxes.extensions.skill_authoring import (
    SkillDrafter,
    ToolSequenceTracker,
    render_skill_template,
)


def _candidate(prompt: str = "set up CI for the project"):
    t = ToolSequenceTracker()
    t.begin_turn(agent_id="coder", user_prompt=prompt)
    t.record_call("Read", {"path": ".github/workflows/ci.yml"})
    t.record_call("Edit", {"path": ".github/workflows/ci.yml", "old": "x", "new": "y"})
    t.record_call("Bash", {"cmd": "pytest"}, status="error", error_message="6 failures")
    t.record_call("Bash", {"cmd": "pytest"}, status="success")
    t.record_call("Write", {"path": "README.md", "content": "..."})
    return t.end_turn(final_response="CI workflow added; tests pass.")


class TestRenderSkillTemplate:
    def test_includes_yaml_frontmatter(self):
        text = render_skill_template(_candidate())
        assert text.startswith("---\n")
        assert "\nname: " in text
        assert "version: 0.1.0" in text

    def test_includes_required_sections(self):
        text = render_skill_template(_candidate())
        assert "# When to use" in text
        assert "# Procedure" in text
        assert "# Pitfalls" in text
        assert "# Verification" in text

    def test_procedure_lists_tools_in_order(self):
        text = render_skill_template(_candidate())
        proc_section = text.split("# Procedure")[1].split("# ")[0]
        assert proc_section.index("Read") < proc_section.index("Edit")
        assert proc_section.index("Edit") < proc_section.index("Bash")

    def test_pitfalls_capture_failure(self):
        text = render_skill_template(_candidate())
        assert "6 failures" in text or "Bash" in text.split("# Pitfalls")[1].split("# ")[0]

    def test_verification_lists_signature(self):
        text = render_skill_template(_candidate())
        verify = text.split("# Verification")[1]
        assert "Read>Edit>Bash>Bash>Write" in verify

    def test_no_pitfalls_when_no_failures(self):
        t = ToolSequenceTracker()
        t.begin_turn(user_prompt="happy path")
        t.record_call("A", {})
        t.record_call("B", {})
        c = t.end_turn()
        text = render_skill_template(c)
        assert "# Pitfalls" not in text

    def test_explicit_name_override(self):
        text = render_skill_template(_candidate(), name="my-custom-name")
        assert "name: my-custom-name" in text

    def test_deterministic(self):
        c = _candidate()
        a = render_skill_template(c)
        b = render_skill_template(c)
        assert a == b


class TestSkillDrafter:
    def test_writes_to_disk(self, tmp_path):
        d = SkillDrafter(tmp_path)
        text, path = d.draft(_candidate())
        assert path is not None
        assert path.exists()
        assert path.name == "SKILL.md"
        assert path.read_text() == text

    def test_no_write_returns_none_path(self, tmp_path):
        d = SkillDrafter(tmp_path)
        text, path = d.draft(_candidate(), write=False)
        assert path is None
        assert "# Procedure" in text

    def test_llm_refinement_applied(self, tmp_path):
        class FakeLLM:
            def complete(self, prompt):
                return "---\nname: refined\nversion: 0.1.0\n---\n# When to use\nbetter\n"

        d = SkillDrafter(tmp_path, llm_client=FakeLLM())
        text, _path = d.draft(_candidate(), write=False)
        assert "name: refined" in text

    def test_llm_failure_falls_back(self, tmp_path):
        class BrokenLLM:
            def complete(self, prompt):
                raise RuntimeError("api down")

        d = SkillDrafter(tmp_path, llm_client=BrokenLLM())
        text, _ = d.draft(_candidate(), write=False)
        assert "# Procedure" in text

    def test_callable_llm_client(self, tmp_path):
        def llm(prompt):
            return "---\nname: cb\nversion: 0.1.0\n---\nbody"

        d = SkillDrafter(tmp_path, llm_client=llm)
        text, _ = d.draft(_candidate(), write=False)
        assert "name: cb" in text

    def test_skills_dir_auto_created(self, tmp_path):
        nested = tmp_path / "deep" / "skills"
        SkillDrafter(nested)
        assert nested.exists()
