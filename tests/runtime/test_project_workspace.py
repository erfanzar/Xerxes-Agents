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

from xerxes.runtime.project_workspace import (
    ensure_project_agent_workspace,
    ensure_project_xerxes_md,
    load_project_agent_workspace,
    project_xerxes_md,
)


def test_ensure_project_agent_workspace_creates_layout(tmp_path):
    created = ensure_project_agent_workspace(tmp_path)

    assert tmp_path / ".agents" in created
    assert (tmp_path / ".agents" / "skills").is_dir()
    assert (tmp_path / ".agents" / "ops" / "OPS.md").is_file()
    assert (tmp_path / ".agents" / "projects" / "README.md").is_file()


def test_load_project_agent_workspace_returns_prompt_context(tmp_path):
    ensure_project_agent_workspace(tmp_path)

    context = load_project_agent_workspace(tmp_path)

    assert context.agents_dir == tmp_path.resolve() / ".agents"
    assert len(context.loaded_files) == 4
    assert "# Project Agent Workspace" in context.prompt
    assert "## .agents/ops/OPS.md" in context.prompt
    assert ".agents/skills/" in context.prompt


def test_ensure_project_xerxes_md_creates_bootstrap_manifest(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo-app"\n', encoding="utf-8")

    path, action = ensure_project_xerxes_md(tmp_path)

    assert action == "created"
    assert path == project_xerxes_md(tmp_path)
    body = path.read_text(encoding="utf-8")
    assert "<!-- XERXES_INIT:BEGIN -->" in body
    assert "<!-- XERXES_INIT:END -->" in body
    assert "Name: demo-app" in body
    assert "Python / uv" in body
    assert "`uv run pytest`" in body
    assert ".agents/skills" in body
    assert "exec_command" in body


def test_ensure_project_xerxes_md_preserves_user_content_and_updates_generated_block(tmp_path):
    path = tmp_path / "XERXES.md"
    path.write_text("# Manual Notes\n\nkeep this\n", encoding="utf-8")

    first_path, first_action = ensure_project_xerxes_md(tmp_path)
    second_path, second_action = ensure_project_xerxes_md(tmp_path)

    assert first_path == path.resolve()
    assert second_path == path.resolve()
    assert first_action == "updated"
    assert second_action == "unchanged"
    body = path.read_text(encoding="utf-8")
    assert body.startswith("# Manual Notes\n\nkeep this")
    assert body.count("<!-- XERXES_INIT:BEGIN -->") == 1
    assert body.count("<!-- XERXES_INIT:END -->") == 1
