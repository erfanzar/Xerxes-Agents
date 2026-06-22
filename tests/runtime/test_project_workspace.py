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

from xerxes.runtime.project_workspace import ensure_project_agent_workspace, load_project_agent_workspace


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
