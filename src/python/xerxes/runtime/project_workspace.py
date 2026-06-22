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

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ..security.prompt_scanner import scan_context_content

PROJECT_AGENTS_DIR = ".agents"
PROJECT_XERXES_FILE = "XERXES.md"
PROJECT_AGENT_CONTEXT_FILES: tuple[str, ...] = (
    "AGENTS.md",
    "SKILL_MAP.md",
    "ops/OPS.md",
    "projects/README.md",
)
_XERXES_INIT_BEGIN = "<!-- XERXES_INIT:BEGIN -->"
_XERXES_INIT_END = "<!-- XERXES_INIT:END -->"

_DEFAULT_FILES: dict[str, str] = {
    "AGENTS.md": """# AGENTS.md

Project-specific instructions for AI agents working in this repository.

## Operating Rules

- Read relevant repository docs and nearby code before changing behavior.
- Use `.agents/ops/OPS.md` for operational runbooks and recurring recovery procedures.
- Use `.agents/projects/` for long-running design, research, migration, or benchmark notes.
- Put reusable task procedures in `.agents/skills/<skill-name>/SKILL.md`.
- Keep claims tied to files, commands, test output, or documented project rules.
""",
    "SKILL_MAP.md": """# Project Skill Map

Map recurring project work to local or shared skills.

| Area | Skill | Notes |
| ---- | ----- | ----- |
| General research/debugging | run-research | Start here for multi-step work. |
| Tests and verification | test-workspace | Adapt this row to the project's real commands. |
| Code review | review-pr | Use for branch or pull-request review. |
""",
    "ops/OPS.md": """# Project Operations

Operational runbooks and environment recovery notes for this project.

Add commands only after verifying they exist in this repository or environment.
Do not store secrets here.
""",
    "projects/README.md": """# Agent Project Notes

Use this directory for long-running design, research, migration, benchmark, or debugging notes.

Suggested note format:

- Goal and stop condition
- Baseline command and result
- Hypotheses tested
- Exact command/output summaries
- Negative results worth preserving
- Next action
""",
}


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


def project_xerxes_md(project_root: str | Path) -> Path:
    """Return ``<project_root>/XERXES.md``."""
    return Path(project_root).expanduser().resolve() / PROJECT_XERXES_FILE


def ensure_project_xerxes_md(project_root: str | Path) -> tuple[Path, str]:
    """Create or refresh the generated project ``XERXES.md`` bootstrap block.

    User-authored content is preserved. If the generated block already exists,
    only that block is replaced; otherwise it is appended to the file.
    """
    root = Path(project_root).expanduser().resolve()
    path = root / PROJECT_XERXES_FILE
    generated = _render_xerxes_md(root)
    block = f"{_XERXES_INIT_BEGIN}\n{generated.rstrip()}\n{_XERXES_INIT_END}\n"
    if not path.exists():
        path.write_text(block, encoding="utf-8")
        return path, "created"

    current = path.read_text(encoding="utf-8")
    if _XERXES_INIT_BEGIN in current and _XERXES_INIT_END in current:
        start = current.index(_XERXES_INIT_BEGIN)
        end = current.index(_XERXES_INIT_END, start) + len(_XERXES_INIT_END)
        updated = current[:start] + block.rstrip() + current[end:]
        if not updated.endswith("\n"):
            updated += "\n"
        if updated != current:
            path.write_text(updated, encoding="utf-8")
            return path, "updated"
        return path, "unchanged"

    separator = "\n\n" if current.strip() else ""
    path.write_text(current.rstrip() + separator + block, encoding="utf-8")
    return path, "updated"


def ensure_project_agent_workspace(project_root: str | Path) -> list[Path]:
    """Create the project-local ``.agents`` layout and seed missing defaults.

    Existing files are left untouched. Returns the paths created in this call.
    """
    agents_dir = project_agents_dir(project_root)
    created: list[Path] = []
    for directory in (agents_dir, agents_dir / "skills", agents_dir / "ops", agents_dir / "projects"):
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created.append(directory)
        else:
            directory.mkdir(parents=True, exist_ok=True)

    for relative, body in _DEFAULT_FILES.items():
        target = agents_dir / relative
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        created.append(target)
    return created


def _render_xerxes_md(root: Path) -> str:
    """Render a compact project bootstrap manifest for model runtime context."""
    name = _project_name(root)
    stack = _detected_stack(root)
    commands = _recommended_commands(root)
    stack_lines = "\n".join(f"- {item}" for item in stack) if stack else "- Unknown; update this after inspection."
    command_lines = "\n".join(f"- `{cmd}`" for cmd in commands) if commands else "- Add verified project commands here."
    return f"""# XERXES.md

Project bootstrap context for Xerxes agents working in this repository.

## Project

- Name: {name}
- Root: `{root}`
- Project agent workspace: `.agents/`

## Detected Stack

{stack_lines}

## Common Commands

{command_lines}

## Runtime Setup

- `/init` maintains this file and the `.agents/` project workspace.
- Project skills live in `.agents/skills/<skill-name>/SKILL.md`.
- Operational runbooks live in `.agents/ops/OPS.md`.
- Long-running investigations and implementation plans live in `.agents/projects/`.
- Durable agent notes and large artifacts should go through project-scoped agent memory, not temporary files.

## Tooling Guidance

- Prefer `exec_command` and `write_stdin` for commands, tests, builds, servers, and interactive processes.
- Use `ExecuteShell` only for short one-shot commands that do not need shell control operators.
- Read files in chunks by default; use full-file reads only when the file is known to be small or intentionally requested.
- Store large tool outputs and agent reports in project memory or project files, then reference the path in the model-visible response.

## Automation Map

- MCP candidates: add repository-specific MCP notes here when servers are configured.
- Skills candidates: promote repeated workflows into `.agents/skills/`.
- Hooks candidates: document pre-commit, CI, formatting, and release hooks here after verifying them.
- Subagent candidates: use coder/researcher/planner/objective agents for parallel work when tasks can be split safely.
"""


def _project_name(root: Path) -> str:
    """Infer a project name from common manifest files."""
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            project = data.get("project", {})
            if isinstance(project, dict):
                name = str(project.get("name") or "").strip()
                if name:
                    return name
            poetry = data.get("tool", {}).get("poetry", {})
            if isinstance(poetry, dict):
                name = str(poetry.get("name") or "").strip()
                if name:
                    return name
        except (OSError, tomllib.TOMLDecodeError):
            pass
    package_json = root / "package.json"
    if package_json.is_file():
        try:
            data = json.loads(package_json.read_text(encoding="utf-8"))
            name = str(data.get("name") or "").strip()
            if name:
                return name
        except (OSError, json.JSONDecodeError):
            pass
    return root.name


def _detected_stack(root: Path) -> list[str]:
    """Return stack hints from common repository markers without executing code."""
    checks = (
        ("Python / uv", "pyproject.toml"),
        ("Node.js", "package.json"),
        ("Rust", "Cargo.toml"),
        ("Go", "go.mod"),
        ("Docker", "Dockerfile"),
        ("Docker Compose", "docker-compose.yml"),
        ("GitHub Actions", ".github/workflows"),
        ("Pre-commit", ".pre-commit-config.yaml"),
        ("Ruff", "ruff.toml"),
        ("Mypy", "mypy.ini"),
        ("Pytest", "pytest.ini"),
    )
    found: list[str] = []
    for label, relative in checks:
        if (root / relative).exists():
            found.append(label)
    return found


def _recommended_commands(root: Path) -> list[str]:
    """Return likely verification commands from manifest files."""
    commands: list[str] = []
    if (root / "pyproject.toml").exists():
        commands.extend(
            [
                "uv run pytest",
                "uv run ruff check .",
                "uv run ruff format --check .",
            ]
        )
    if (root / "package.json").exists():
        commands.append("npm test")
    if (root / "Cargo.toml").exists():
        commands.append("cargo test")
    if (root / "go.mod").exists():
        commands.append("go test ./...")
    if (root / ".pre-commit-config.yaml").exists():
        commands.append("uv run pre-commit run --all-files")
    return commands


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
    "PROJECT_XERXES_FILE",
    "ProjectAgentWorkspaceContext",
    "ensure_project_agent_workspace",
    "ensure_project_xerxes_md",
    "load_project_agent_workspace",
    "project_agent_skills_dir",
    "project_agents_dir",
    "project_xerxes_md",
]
