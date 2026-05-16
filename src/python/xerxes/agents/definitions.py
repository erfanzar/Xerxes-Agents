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
"""Agent definition registry and loader.

This module manages built-in and user-defined agent definitions, loading them
from YAML files, Markdown files with frontmatter, and project-level
``agents.yaml`` configurations.

Main exports:
    - AgentDefinition: Dataclass representing a loaded agent definition.
    - BUILTIN_AGENTS: Dictionary of built-in agent definitions.
    - load_agent_definitions: Load all available agent definitions.
    - get_agent_definition: Retrieve a single definition by name.
    - list_agent_definitions: List all loaded definitions sorted by name.
    - list_agent_definition_load_errors: Retrieve load-time errors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentDefinition:
    """A loaded agent definition with metadata and configuration.

    Attributes:
        name (str): Unique agent identifier.
        description (str): Human-readable description.
        system_prompt (str): System prompt text.
        model (str): Default model identifier.
        tools (list[str]): Tool names this agent uses.
        allowed_tools (list[str] | None): Whitelist of allowed tools.
        exclude_tools (list[str]): Tools explicitly excluded.
        source (str): Origin (e.g., ``"built-in"``, ``"user"``, ``"project"``).
        max_depth (int): Maximum sub-agent delegation depth.
        isolation (str): Isolation strategy string.
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


BUILTIN_AGENTS_DIR = Path(__file__).parent / "default"


def _load_builtin_agents() -> dict[str, AgentDefinition]:
    """Load built-in definitions from ``default/``; fall back to hardcoded set."""
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

    if not defs:
        defs = _HARDCODED_BUILTIN_AGENTS
    return defs


BUILTIN_AGENTS: dict[str, AgentDefinition] = {}
_LAST_LOAD_ERRORS: list[str] = []

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
        allowed_tools=["ReadFile", "GlobTool", "GrepTool", "ListDir"],
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
        allowed_tools=["ReadFile", "GlobTool", "GrepTool", "ListDir", "DuckDuckGoSearch"],
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
        allowed_tools=["ReadFile", "GlobTool", "GrepTool", "ListDir"],
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

BUILTIN_AGENTS = _load_builtin_agents()


def _parse_agent_md(path: Path, source: str = "user") -> AgentDefinition:
    """Parse a Markdown agent file with optional ``---`` YAML frontmatter.

    Frontmatter pulls ``description``, ``model``, ``tools``, ``max_depth``
    and ``isolation``; the rest of the file becomes the system prompt. The
    agent name is the file stem. ``source`` is stamped onto the returned
    definition unchanged.
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
    """Parse YAML frontmatter, with a key:value line fallback if PyYAML is missing."""
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
    """Merge built-in, user, and project agent definitions into one mapping.

    Precedence is built-in < user < project — later entries override earlier
    ones by ``name``. Per-file load errors are captured in
    ``_LAST_LOAD_ERRORS`` (see :func:`list_agent_definition_load_errors`)
    rather than raised.

    Args:
        user_dir: Override for the user agents directory; defaults to the
            ``agents`` subdir of the xerxes home.
        project_dir: Override for the project agents directory; defaults to
            ``.xerxes/agents`` under the current working directory.
    """
    global _LAST_LOAD_ERRORS
    _LAST_LOAD_ERRORS = []
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
            except Exception as exc:
                _record_load_error(p, exc)
        for p in sorted(udir.glob("*.md")):
            try:
                d = _parse_agent_md(p, source="user")
                defs[d.name] = d
            except Exception as exc:
                _record_load_error(p, exc)

    pdir = project_dir or Path.cwd() / ".xerxes" / "agents"
    if pdir.is_dir():
        for p in sorted(pdir.glob("*.yaml")):
            try:
                d = _parse_agent_yaml(p, source="project")
                if d:
                    defs[d.name] = d
            except Exception as exc:
                _record_load_error(p, exc)
        for p in sorted(pdir.glob("*.md")):
            try:
                d = _parse_agent_md(p, source="project")
                defs[d.name] = d
            except Exception as exc:
                _record_load_error(p, exc)

    for p, source in _project_agent_candidates(Path.cwd()):
        try:
            for d in _parse_project_agent_file(p, source=source):
                defs[d.name] = d
        except Exception as exc:
            _record_load_error(p, exc)

    return defs


def _record_load_error(path: Path, exc: Exception) -> None:
    """Append a formatted entry to ``_LAST_LOAD_ERRORS`` and log a warning."""
    message = f"{path}: {type(exc).__name__}: {exc}"
    _LAST_LOAD_ERRORS.append(message)
    logger.warning("Failed to load agent definition %s: %s", path, exc)


def _project_agent_candidates(cwd: Path) -> list[tuple[Path, str]]:
    """Return existing ``(path, source)`` pairs for project-level spec files."""
    candidates = [
        (cwd / ".kimi" / "agent.yaml", "project"),
        (cwd / ".kimi" / "agents.yaml", "project"),
        (cwd / "agent.yaml", "project"),
        (cwd / "agents.yaml", "project"),
    ]
    return [(path, source) for path, source in candidates if path.is_file()]


def _parse_project_agent_file(path: Path, source: str) -> list[AgentDefinition]:
    """Parse a single ``agent.yaml`` or a multi-agent ``agents.yaml``.

    For ``agents.yaml`` each child of the top-level ``agents:`` mapping
    becomes its own normalised spec; for any other filename a single spec
    is returned. ``source`` is stamped onto every returned definition.

    Raises:
        RuntimeError: If PyYAML is not installed.
        ValueError: If ``agents`` is present but not a mapping.
    """
    if path.name != "agents.yaml":
        parsed = _parse_agent_yaml(path, source=source)
        return [parsed] if parsed else []

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load agents.yaml files") from exc

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "agents" not in raw:
        parsed = _parse_agent_yaml(path, source=source)
        return [parsed] if parsed else []

    if not isinstance(raw["agents"], dict):
        raise ValueError("agents.yaml field 'agents' must be a mapping")

    defs: list[AgentDefinition] = []
    for name, body in raw["agents"].items():
        if not isinstance(body, dict):
            raise ValueError(f"agents.{name} must be a mapping")
        normalized = {
            "version": str(raw.get("version", "1")),
            "agent": {"name": str(name), **body},
        }
        temp_path = path.with_name(f".{path.stem}.{name}.yaml")
        spec = _load_agent_spec_from_data(temp_path, normalized)
        defs.append(_agent_definition_from_spec(spec, source))
    return defs


def _load_agent_spec_from_data(path: Path, data: dict[str, Any]):
    """Round-trip ``data`` through a tempfile so :func:`load_agent_spec` can read it.

    Used by the ``agents.yaml`` parser to reuse the regular loading
    pipeline (including ``extend`` resolution) for inline agent
    definitions. The tempfile is removed unconditionally.
    """
    import tempfile

    import yaml

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", dir=path.parent, delete=False) as f:
        yaml.safe_dump(data, f, sort_keys=False)
        temp_path = Path(f.name)
    try:
        from .agentspec import load_agent_spec

        return load_agent_spec(temp_path)
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass


def _agent_definition_from_spec(spec: Any, source: str) -> AgentDefinition:
    """Project a :class:`ResolvedAgentSpec` into the registry's :class:`AgentDefinition`."""
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


def _parse_agent_yaml(path: Path, source: str = "user") -> AgentDefinition | None:
    """Load ``path`` via :func:`load_agent_spec` and project to :class:`AgentDefinition`."""
    from .agentspec import load_agent_spec

    spec = load_agent_spec(path)
    return _agent_definition_from_spec(spec, source)


def list_agent_definition_load_errors() -> list[str]:
    """Return formatted error strings from the last :func:`load_agent_definitions` call.

    Triggers a reload if the cache is empty so callers asking right after
    process startup get a non-stale answer.
    """
    if not _LAST_LOAD_ERRORS:
        load_agent_definitions()
    return list(_LAST_LOAD_ERRORS)


def get_agent_definition(name: str) -> AgentDefinition | None:
    """Return the definition registered as ``name``, or ``None`` if missing."""
    return load_agent_definitions().get(name)


def list_agent_definitions() -> list[AgentDefinition]:
    """Return every loaded definition sorted by name."""
    return sorted(load_agent_definitions().values(), key=lambda d: d.name)


__all__ = [
    "BUILTIN_AGENTS",
    "AgentDefinition",
    "get_agent_definition",
    "list_agent_definition_load_errors",
    "list_agent_definitions",
    "load_agent_definitions",
]
