# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""Agent definitions with built-in types and file-based loading.

Inspired by the nano-claude-code ``AgentDefinition`` pattern, this module
provides:

- **Built-in agent types**: general-purpose, coder, reviewer, researcher,
  tester, planner, data-analyst — ready to use out of the box.
- **File-based definitions**: Load agent definitions from ``.md`` files
  with YAML frontmatter, supporting user-level (``~/.xerxes/agents/``)
  and project-level (``.xerxes/agents/``) overrides.
- **Registry**: A unified lookup for all agent definitions.

File format::

    ---
    description: "Short description"
    model: claude-sonnet-4-6
    tools: [Read, Write, Edit, Bash]
    ---

    System prompt body goes here...

Usage::

    from xerxes.agents.definitions import (
        get_agent_definition,
        load_agent_definitions,
        BUILTIN_AGENTS,
    )


    coder = get_agent_definition("coder")


    all_defs = load_agent_definitions()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentDefinition:
    """Definition for a specialized agent type.

    Attributes:
        name: Canonical agent name (e.g. ``"coder"``, ``"reviewer"``).
        description: Human-readable description of the agent's purpose.
        system_prompt: Extra instructions prepended to the base system prompt.
        model: Model override. Empty string means inherit from parent.
        tools: Tool name whitelist. Empty list means all tools allowed.
        source: Origin — ``"built-in"``, ``"user"``, or ``"project"``.
        max_depth: Maximum nesting depth for sub-agent spawning.
        isolation: Default isolation mode (``""`` or ``"worktree"``).
    """

    name: str
    description: str = ""
    system_prompt: str = ""
    model: str = ""
    tools: list[str] = field(default_factory=list)
    source: str = "built-in"
    max_depth: int = 5
    isolation: str = ""


BUILTIN_AGENTS: dict[str, AgentDefinition] = {
    "general-purpose": AgentDefinition(
        name="general-purpose",
        description=(
            "General-purpose agent for researching complex questions, "
            "searching for code, and executing multi-step tasks."
        ),
        system_prompt="",
        source="built-in",
    ),
    "coder": AgentDefinition(
        name="coder",
        description="Specialized coding agent for writing, reading, and modifying code.",
        system_prompt=(
            "You are a specialized coding assistant. Focus on:\n"
            "- Writing clean, idiomatic code\n"
            "- Reading and understanding existing code before modifying\n"
            "- Making minimal targeted changes\n"
            "- Never adding unnecessary features, comments, or error handling\n"
        ),
        source="built-in",
    ),
    "reviewer": AgentDefinition(
        name="reviewer",
        description="Code review agent analyzing quality, security, and correctness.",
        system_prompt=(
            "You are a code reviewer. Analyze code for:\n"
            "- Correctness and logic errors\n"
            "- Security vulnerabilities (injection, XSS, auth bypass, etc.)\n"
            "- Performance issues\n"
            "- Code quality and maintainability\n"
            "Be concise and specific. Categorize findings as: Critical | Warning | Suggestion.\n"
        ),
        tools=["Read", "ReadFile", "Glob", "Grep", "ListDir"],
        source="built-in",
    ),
    "researcher": AgentDefinition(
        name="researcher",
        description="Research agent for exploring codebases and answering questions.",
        system_prompt=(
            "You are a research assistant focused on understanding codebases.\n"
            "- Read and analyze code thoroughly before answering\n"
            "- Provide factual, evidence-based answers\n"
            "- Cite specific file paths and line numbers\n"
            "- Be concise and focused\n"
        ),
        tools=["Read", "ReadFile", "Glob", "Grep", "ListDir", "GoogleSearch"],
        source="built-in",
    ),
    "tester": AgentDefinition(
        name="tester",
        description="Testing agent that writes and runs tests.",
        system_prompt=(
            "You are a testing specialist. Your job:\n"
            "- Write comprehensive tests for the given code\n"
            "- Run existing tests and diagnose failures\n"
            "- Focus on edge cases and error conditions\n"
            "- Keep tests simple, readable, and fast\n"
        ),
        source="built-in",
    ),
    "planner": AgentDefinition(
        name="planner",
        description="Planning agent that designs implementation strategies and task breakdowns.",
        system_prompt=(
            "You are an expert software architect and planner.\n"
            "- Break complex tasks into clear, actionable steps\n"
            "- Identify dependencies and critical paths\n"
            "- Consider trade-offs and alternatives\n"
            "- Produce structured plans, not code\n"
        ),
        tools=["Read", "ReadFile", "Glob", "Grep", "ListDir"],
        source="built-in",
    ),
    "data-analyst": AgentDefinition(
        name="data-analyst",
        description="Data analysis agent for processing and analyzing data.",
        system_prompt=(
            "You are a data analysis specialist.\n"
            "- Process and analyze data efficiently\n"
            "- Use appropriate statistical methods\n"
            "- Present findings clearly with summaries\n"
            "- Handle various data formats (JSON, CSV, etc.)\n"
        ),
        source="built-in",
    ),
}


def _parse_agent_md(path: Path, source: str = "user") -> AgentDefinition:
    """Parse a ``.md`` file with optional YAML frontmatter into an AgentDefinition.

    File format::

        ---
        description: "Short description"
        model: claude-sonnet-4-6
        tools: [Read, Write, Edit, Bash]
        max_depth: 3
        isolation: worktree
        ---

        System prompt body goes here...
    """
    content = path.read_text()
    name = path.stem
    description = ""
    model = ""
    tools: list[str] = []
    max_depth = 5
    isolation = ""
    system_prompt_body = content

    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            fm_text = content[3:end].strip()
            system_prompt_body = content[end + 3 :].strip()

            fm = _parse_frontmatter(fm_text)
            description = str(fm.get("description", ""))
            model = str(fm.get("model", ""))
            max_depth = int(fm.get("max_depth", 5))
            isolation = str(fm.get("isolation", ""))

            raw_tools = fm.get("tools", [])
            if isinstance(raw_tools, list):
                tools = [str(t) for t in raw_tools]
            elif isinstance(raw_tools, str):
                s = raw_tools.strip("[]")
                tools = [t.strip() for t in s.split(",") if t.strip()]

    return AgentDefinition(
        name=name,
        description=description,
        system_prompt=system_prompt_body,
        model=model,
        tools=tools,
        source=source,
        max_depth=max_depth,
        isolation=isolation,
    )


def _parse_frontmatter(text: str) -> dict[str, Any]:
    """Parse YAML frontmatter, falling back to manual key:value parsing."""
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except ImportError:
        pass

    fm: dict[str, Any] = {}
    for line in text.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            key = k.strip()
            val = v.strip()
            if val.startswith("[") and val.endswith("]"):
                items = val[1:-1].split(",")
                fm[key] = [item.strip().strip("'\"") for item in items if item.strip()]
            else:
                fm[key] = val.strip("'\"")
    return fm


def load_agent_definitions(
    user_dir: Path | None = None,
    project_dir: Path | None = None,
) -> dict[str, AgentDefinition]:
    """Load all agent definitions: built-ins, then user-level, then project-level.

    Search paths:

    - Built-in definitions (always loaded first).
    - ``~/.xerxes/agents/*.md`` (user-level, overrides built-ins).
    - ``.xerxes/agents/*.md`` (project-level, overrides user).

    Args:
        user_dir: Override user-level directory.
        project_dir: Override project-level directory.

    Returns:
        Dict mapping agent names to their definitions.
    """
    defs: dict[str, AgentDefinition] = dict(BUILTIN_AGENTS)

    if user_dir is None:
        from xerxes.core.paths import xerxes_subdir

        udir = xerxes_subdir("agents")
    else:
        udir = user_dir
    if udir.is_dir():
        for p in sorted(udir.glob("*.md")):
            try:
                d = _parse_agent_md(p, source="user")
                defs[d.name] = d
            except Exception:
                pass

    pdir = project_dir or Path.cwd() / ".xerxes" / "agents"
    if pdir.is_dir():
        for p in sorted(pdir.glob("*.md")):
            try:
                d = _parse_agent_md(p, source="project")
                defs[d.name] = d
            except Exception:
                pass

    return defs


def get_agent_definition(name: str) -> AgentDefinition | None:
    """Look up an agent definition by name.

    Args:
        name: Agent definition name (e.g. ``"coder"``, ``"reviewer"``).

    Returns:
        The :class:`AgentDefinition`, or ``None`` if not found.
    """
    return load_agent_definitions().get(name)


def list_agent_definitions() -> list[AgentDefinition]:
    """Return all available agent definitions sorted by name."""
    return sorted(load_agent_definitions().values(), key=lambda d: d.name)


__all__ = [
    "BUILTIN_AGENTS",
    "AgentDefinition",
    "get_agent_definition",
    "list_agent_definitions",
    "load_agent_definitions",
]
