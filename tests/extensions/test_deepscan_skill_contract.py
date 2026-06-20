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
"""DeepScan skill contract regressions."""

from __future__ import annotations

from pathlib import Path


def test_deepscan_skill_requires_project_memory_chunked_reads_and_no_tool_cap() -> None:
    skill = Path("src/python/xerxes/skills/deepscan/SKILL.md").read_text(encoding="utf-8")
    lowered = skill.lower()

    assert "project-scoped agent memory" in lowered
    assert 'agent_memory_write("project", "deepscan/agent_notes.md"' in lowered
    assert "do not fall back to `tmp-files`" in lowered
    assert "readfile(file_path=..., offset=..., limit=...)" in lowered
    assert "`limit=-1` only when the whole file is intentionally required" in lowered
    assert "no artificial tool-call cap" in lowered
    assert "≤15" not in skill
    assert "<=15" not in skill
    assert "15 tool" not in lowered
