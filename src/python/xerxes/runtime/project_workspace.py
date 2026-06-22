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
"""Project-local agent workspace helpers.

Xerxes has private runtime memory under ``~/.xerxes/projects/<hash>/memory``.
This module handles the complementary project-owned layout that should travel
with a repository:

```
.agents/
  AGENTS.md
  SKILL_MAP.md
  skills/
  ops/OPS.md
  projects/README.md
```

The runtime treats this as structured workspace context, not as hidden memory:
it is visible in the repo, versionable if the project wants that, and loaded as
a compact index before each model turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..security.prompt_scanner import scan_context_content

PROJECT_AGENTS_DIR = ".agents"
PROJECT_AGENT_CONTEXT_FILES: tuple[str, ...] = (
    "AGENTS.md",
    "SKILL_MAP.md",
    "ops/OPS.md",
    "projects/README.md",
)


@dataclass(frozen=True)
class ProjectAgentWorkspaceContext:
    """Rendered project-local agent workspace context.

    Attributes:
        root: Repository root that owns the ``.agents`` directory.
        agents_dir: Absolute path to ``root/.agents``.
        prompt: Sanitized prompt block, empty when no project workspace exists.
        loaded_files: Files that contributed to ``prompt``.
    """

    root: Path
    agents_dir: Path
    prompt: str
    loaded_files: tuple[Path, ...]


def project_agents_dir(project_root: str | Path) -> Path:
    """Return ``<project_root>/.agents``."""
    return Path(project_root).expanduser().resolve() / PROJECT_AGENTS_DIR


def project_agent_skills_dir(project_root: str | Path) -> Path:
    """Return ``<project_root>/.agents/skills``."""
    return project_agents_dir(project_root) / "skills"


def load_project_agent_workspace(
    project_root: str | Path,
    *,
    max_bytes_per_file: int = 6_000,
) -> ProjectAgentWorkspaceContext:
    """Load the visible project-local ``.agents`` context if present.

    Missing files are skipped. File bodies are bounded before prompt injection
    to keep this section an index/runbook, not a transcript dump.
    """
    root = Path(project_root).expanduser().resolve()
    agents_dir = root / PROJECT_AGENTS_DIR
    if not agents_dir.is_dir():
        return ProjectAgentWorkspaceContext(root=root, agents_dir=agents_dir, prompt="", loaded_files=())

    parts = [
        "# Project Agent Workspace",
        f"Directory: {agents_dir}",
        "",
        "This is project-owned agent operating context. Use it to stay organized:",
        "- `.agents/skills/` contains repository-local skills.",
        "- `.agents/ops/OPS.md` contains operational runbooks.",
        "- `.agents/projects/` contains long-running project notes.",
        "",
        "Read the referenced files with normal file tools when the task needs more detail.",
    ]
    loaded: list[Path] = []
    for relative in PROJECT_AGENT_CONTEXT_FILES:
        path = agents_dir / relative
        if not path.is_file():
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if not raw.strip():
            continue
        clipped = raw[:max_bytes_per_file]
        if len(raw) > max_bytes_per_file:
            clipped += "\n\n[truncated: read this file directly for the rest]"
        safe = scan_context_content(clipped, filename=f"Project agent workspace: {path}")
        if "[BLOCKED:" in safe:
            continue
        loaded.append(path)
        parts.extend(["", f"## .agents/{relative}", safe.strip()])

    prompt = "\n".join(parts).strip() if loaded else ""
    return ProjectAgentWorkspaceContext(root=root, agents_dir=agents_dir, prompt=prompt, loaded_files=tuple(loaded))


__all__ = [
    "PROJECT_AGENTS_DIR",
    "PROJECT_AGENT_CONTEXT_FILES",
    "ProjectAgentWorkspaceContext",
    "load_project_agent_workspace",
    "project_agent_skills_dir",
    "project_agents_dir",
]
