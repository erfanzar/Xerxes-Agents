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

# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");

from xerxes.runtime.project_workspace import load_project_agent_workspace


def test_load_project_agent_workspace_returns_prompt_context(tmp_path):
    agents_dir = tmp_path / ".agents"
    (agents_dir / "ops").mkdir(parents=True)
    (agents_dir / "projects").mkdir()
    (agents_dir / "AGENTS.md").write_text("# Project Agents\n\nFollow local rules.\n", encoding="utf-8")
    (agents_dir / "SKILL_MAP.md").write_text("# Skill Map\n\nUse project skills.\n", encoding="utf-8")
    (agents_dir / "ops" / "OPS.md").write_text("# Ops\n\nRun verified commands only.\n", encoding="utf-8")
    (agents_dir / "projects" / "README.md").write_text("# Project Notes\n\nKeep notes here.\n", encoding="utf-8")

    context = load_project_agent_workspace(tmp_path)

    assert context.agents_dir == tmp_path.resolve() / ".agents"
    assert len(context.loaded_files) == 4
    assert "# Project Agent Workspace" in context.prompt
    assert "## .agents/ops/OPS.md" in context.prompt
    assert ".agents/skills/" in context.prompt
