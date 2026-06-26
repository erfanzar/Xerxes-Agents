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
"""Contract tests for the bundled Pallas kernel skill."""

from __future__ import annotations

from pathlib import Path

from xerxes.extensions.skills import SkillRegistry, parse_skill_md


def _skill_path() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "python" / "xerxes" / "skills" / "pallas-kernel"


def test_pallas_kernel_skill_parses_contract() -> None:
    skill_md = _skill_path() / "SKILL.md"
    skill = parse_skill_md(skill_md.read_text(), skill_md)

    assert skill.name == "pallas-kernel"
    assert "Pallas" in skill.metadata.description
    assert "MaxKernel" in skill.instructions
    assert "references/prompt-pack.md" in skill.instructions
    assert "debug=True" in skill.instructions
    assert "limit=-1" in skill.instructions
    assert "ExecuteShell" not in skill.metadata.required_tools
    assert {"ReadFile", "WriteFile", "FileEditTool", "GrepTool", "GlobTool", "exec_command", "write_stdin"}.issubset(
        set(skill.metadata.required_tools)
    )
    assert {"auto", "plan", "implement", "validate", "test", "profile", "autotune", "explain", "gpu-to-jax"} == set(
        skill.metadata.subcommands
    )


def test_pallas_kernel_skill_discovers_with_reference_workflows() -> None:
    registry = SkillRegistry()
    registry.discover(_skill_path().parent)

    skill = registry.get("pallas-kernel")
    assert skill is not None
    references = skill.source_path.parent / "references"
    workflow_names = {path.name for path in references.glob("*-workflow.md")}
    prompt_pack = references / "prompt-pack.md"
    knowledge_base = references / "knowledge-base.md"
    subagents = references / "subagents.md"

    assert workflow_names == {
        "auto-workflow.md",
        "autotune-workflow.md",
        "explain-workflow.md",
        "gpu-to-jax-workflow.md",
        "implement-workflow.md",
        "plan-workflow.md",
        "profile-workflow.md",
        "test-workflow.md",
        "validate-workflow.md",
    }
    prompt_text = prompt_pack.read_text()
    assert "Kernel Planner" in prompt_text
    assert "Kernel Implementer" in prompt_text
    assert "Compilation Fixer" in prompt_text
    assert "Test Validator" in prompt_text
    assert "Autotuner" in prompt_text
    assert "Pipeline Coordinator" in prompt_text
    knowledge_text = knowledge_base.read_text()
    assert "Pallas programming model" in knowledge_text
    assert "Mosaic TPU constraints" in knowledge_text
    assert "TPU specs reference" in knowledge_text
    subagent_text = subagents.read_text()
    assert "AutonomousPipelineAgent" in subagent_text
    assert "ValidateKernelCompilationAgent" in subagent_text
    assert "ProfileAgentOrchestrator" in subagent_text
