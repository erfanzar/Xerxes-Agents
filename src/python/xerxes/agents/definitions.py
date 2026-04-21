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
- **YAML agent specs**: ``agent.yaml`` files with inheritance (Kimi-style).
- **Registry**: A unified lookup for all agent definitions.

File format::

    ---
    description: "Short description"
    model: claude-sonnet-4-6
    tools: [Read, Write, Edit, Bash]
    ---

    System prompt body goes here...

Or as a YAML spec (``agent.yaml``)::

    version: 1
    agent:
      name: coder
      extend: default
      system_prompt_path: ./system.md
      allowed_tools:
        - ReadFile
        - WriteFile

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

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentDefinition:
    """Definition for a specialized agent type.

    Attributes:
        name: Canonical agent name (e.g. ``"coder"``, ``"reviewer"``).
        description: Human-readable description of the agent's purpose.
        system_prompt: Extra instructions prepended to the base system prompt.
        model: Model override. Empty string means inherit from parent.
        tools: Tool name whitelist. Empty list means all tools allowed.
        allowed_tools: Like ``tools`` but intended for additive filtering.
        exclude_tools: Tool names to explicitly remove from the available set.
        source: Origin — ``"built-in"``, ``"user"``, ``"project"``, or ``"yaml"``.
        max_depth: Maximum nesting depth for sub-agent spawning.
        isolation: Default isolation mode (``""`` or ``"worktree"``).
    """

    name: str
    description: str = ""
    system_prompt: str = ""
    model: str = ""
    tools: list[str] = field(default_factory=list)
    allowed_tools: list[str] | None = None
    exclude_tools: list[str] = field(default_factory=list)
    source: str = "built-in"
    max_depth: int = 5
    isolation: str = ""


# Directory where built-in YAML agent specs live.
BUILTIN_AGENTS_DIR = Path(__file__).parent / "default"


def _load_builtin_agents() -> dict[str, AgentDefinition]:
    """Load built-in agents from YAML specs if available, else fall back to hard-coded."""
    defs: dict[str, AgentDefinition] = {}
    if BUILTIN_AGENTS_DIR.is_dir():
        for yaml_path in sorted(BUILTIN_AGENTS_DIR.glob("*.yaml")):
            try:
                from .agentspec import load_agent_spec

                spec = load_agent_spec(yaml_path)
                defs[spec.name] = AgentDefinition(
                    name=spec.name,
                    description=spec.when_to_use,
                    system_prompt=spec.system_prompt,
                    model=spec.model or "",
                    tools=spec.tools,
                    allowed_tools=spec.allowed_tools,
                    exclude_tools=spec.exclude_tools,
                    source="built-in",
                    max_depth=spec.max_depth,
                    isolation=spec.isolation,
                )
            except Exception as exc:
                logger.debug("Failed to load built-in agent spec %s: %s", yaml_path, exc)

    # Fallback hard-coded definitions if YAML loading fails or dir is missing
    if not defs:
        defs = _HARDCODED_BUILTIN_AGENTS
    return defs


# Module-level constant: populated once at import time.
BUILTIN_AGENTS: dict[str, AgentDefinition] = {}

# Hard-coded fallback definitions (used when YAML specs are not present).
_HARDCODED_BUILTIN_AGENTS: dict[str, AgentDefinition] = {
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
        allowed_tools=["ReadFile", "Glob", "Grep", "ListDir"],
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
        allowed_tools=["ReadFile", "Glob", "Grep", "ListDir", "GoogleSearch"],
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
        allowed_tools=["ReadFile", "Glob", "Grep", "ListDir"],
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

# Populate built-in agents from YAML specs (falls back to hard-coded above).
BUILTIN_AGENTS = _load_builtin_agents()


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
    - ``~/.xerxes/agents/*.yaml`` and ``*.md`` (user-level, overrides built-ins).
    - ``.xerxes/agents/*.yaml`` and ``*.md`` (project-level, overrides user).

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
        for p in sorted(udir.glob("*.yaml")):
            try:
                d = _parse_agent_yaml(p, source="user")
                if d:
                    defs[d.name] = d
            except Exception:
                pass
        for p in sorted(udir.glob("*.md")):
            try:
                d = _parse_agent_md(p, source="user")
                defs[d.name] = d
            except Exception:
                pass

    pdir = project_dir or Path.cwd() / ".xerxes" / "agents"
    if pdir.is_dir():
        for p in sorted(pdir.glob("*.yaml")):
            try:
                d = _parse_agent_yaml(p, source="project")
                if d:
                    defs[d.name] = d
            except Exception:
                pass
        for p in sorted(pdir.glob("*.md")):
            try:
                d = _parse_agent_md(p, source="project")
                defs[d.name] = d
            except Exception:
                pass

    return defs


def _parse_agent_yaml(path: Path, source: str = "user") -> AgentDefinition | None:
    """Parse an ``agent.yaml`` file into an :class:`AgentDefinition`."""
    from .agentspec import load_agent_spec

    spec = load_agent_spec(path)
    return AgentDefinition(
        name=spec.name,
        description=spec.when_to_use,
        system_prompt=spec.system_prompt,
        model=spec.model or "",
        tools=spec.tools,
        allowed_tools=spec.allowed_tools,
        exclude_tools=spec.exclude_tools,
        source=source,
        max_depth=spec.max_depth,
        isolation=spec.isolation,
    )


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
