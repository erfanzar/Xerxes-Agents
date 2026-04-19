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


"""Skill discovery and management for Xerxes.

Skills are self-contained instruction packages defined by SKILL.md files.
Each skill provides metadata, instructions, and optional resource references
that can be injected into agent prompts on demand.

Skill directory layout::

    skills/
    ├── web_research/
    │   └── SKILL.md
    ├── code_review/
    │   └── SKILL.md
    └── data_analysis/
        ├── SKILL.md
        └── templates/
            └── report.md

SKILL.md format (YAML frontmatter + markdown body)::

    ---
    name: web_research
    description: Search the web and synthesize findings
    version: "1.0"
    tags: [research, web]
    resources:
      - templates/query.md
    ---



    Instructions for conducting web research...
"""

from __future__ import annotations

import logging
import re
import sys
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_PLATFORM_MAP = {
    "macos": "darwin",
    "linux": "linux",
    "windows": "win32",
}


@dataclass
class SkillMetadata:
    """Parsed metadata from the YAML frontmatter of a SKILL.md file.

    Attributes:
        name: A unique identifier for the skill (e.g., ``"web_research"``).
        description: A short human-readable description of what the skill does.
        version: A version string for the skill (default ``"1.0"``).
        tags: Freeform tags used for search and categorization
            (e.g., ``["research", "web"]``).
        resources: Relative paths to supplementary files (templates, data)
            located alongside the SKILL.md file.
        author: The name or handle of the skill author.
        dependencies: Names of other skills that must be loaded before this
            skill can function.
        required_tools: Names of tools (from the plugin registry) that this
            skill expects to be available at runtime.
        platforms: OS platforms this skill supports (e.g. ``["macos", "linux"]``).
            Empty means all platforms.
        config_vars: Skill-declared config keys that the user can set in
            ``~/.xerxes/config.yaml`` under ``skills.config.<skill_name>``.
        trust_level: ``"builtin"``, ``"trusted"``, or ``"community"``.
        source: Where the skill came from: ``"local"``, ``"github"``, ``"hub"``.
        setup_command: Optional shell command to run before first use.
    """

    name: str
    description: str = ""
    version: str = "1.0"
    tags: list[str] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)
    author: str = ""
    dependencies: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)
    config_vars: list[str] = field(default_factory=list)
    trust_level: str = "community"
    source: str = "local"
    setup_command: str = ""


@dataclass
class Skill:
    """A fully loaded skill consisting of metadata, instructions, and resource references.

    Attributes:
        metadata: The :class:`SkillMetadata` parsed from the SKILL.md
            frontmatter.
        instructions: The markdown body of the SKILL.md file, containing
            the instructions to be injected into the agent prompt.
        source_path: The filesystem path to the SKILL.md file this skill
            was loaded from.
        resources_dir: The directory containing supplementary resource files,
            or ``None`` if no resources are declared.
    """

    metadata: SkillMetadata
    instructions: str
    source_path: Path
    resources_dir: Path | None = None

    @property
    def name(self) -> str:
        """Return the skill's unique name from its metadata.

        Returns:
            The skill name string.
        """
        return self.metadata.name

    def to_prompt_section(self) -> str:
        """Format the skill as a markdown section for injection into a system prompt.

        Produces a section with a ``
        description line, and the full instruction body.

        Returns:
            A markdown-formatted string ready to be appended to a system
            prompt.
        """
        header = f"## Skill: {self.metadata.name}"
        if self.metadata.description:
            header += f"\n{self.metadata.description}"
        return f"{header}\n\n{self.instructions}"


def parse_skill_md(content: str, source_path: Path) -> Skill:
    """Parse a SKILL.md file's content into a :class:`Skill` object.

    The file is expected to contain optional YAML frontmatter delimited by
    ``---`` lines, followed by a markdown body.  If PyYAML is installed the
    frontmatter is parsed with ``yaml.safe_load``; otherwise a simple
    line-by-line key-value parser is used as a fallback.

    When no ``name`` key is present in the frontmatter, the parent directory
    name of *source_path* is used as the skill name.

    Args:
        content: The full text content of the SKILL.md file.
        source_path: The filesystem path to the SKILL.md file (used to
            derive the skill name and resources directory).

    Returns:
        A :class:`Skill` instance populated with the parsed metadata and
        instruction body.

    Example:
        >>> from pathlib import Path
        >>> content = "---\\nname: demo\\n---\\nDo something."
        >>> skill = parse_skill_md(content, Path("/skills/demo/SKILL.md"))
        >>> skill.name
        'demo'
    """
    metadata_dict: dict = {}
    body = content

    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", content, re.DOTALL)
    if fm_match:
        fm_text = fm_match.group(1)
        body = fm_match.group(2).strip()

        try:
            import yaml

            metadata_dict = yaml.safe_load(fm_text) or {}
        except ImportError:
            for line in fm_text.strip().splitlines():
                line = line.strip()
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if value.startswith("[") and value.endswith("]"):
                        value = [v.strip().strip('"').strip("'") for v in value[1:-1].split(",")]
                    metadata_dict[key] = value

    name = metadata_dict.get("name", source_path.parent.name)

    def _normalize_list(value):
        """Coerce str, list, or None to a clean list of strings."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()]
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return []

    metadata = SkillMetadata(
        name=name,
        description=metadata_dict.get("description", ""),
        version=str(metadata_dict.get("version", "1.0")),
        tags=_normalize_list(metadata_dict.get("tags")),
        resources=_normalize_list(metadata_dict.get("resources")),
        author=metadata_dict.get("author", ""),
        dependencies=_normalize_list(metadata_dict.get("dependencies")),
        required_tools=_normalize_list(metadata_dict.get("required_tools")),
        platforms=_normalize_list(metadata_dict.get("platforms")),
        config_vars=_normalize_list(metadata_dict.get("config_vars")),
        trust_level=str(metadata_dict.get("trust_level", "community")),
        source=str(metadata_dict.get("source", "local")),
        setup_command=metadata_dict.get("setup_command", ""),
    )

    resources_dir = source_path.parent if metadata.resources else None

    return Skill(
        metadata=metadata,
        instructions=body,
        source_path=source_path,
        resources_dir=resources_dir,
    )


class SkillRegistry:
    """Discovers, indexes, and provides skills for prompt injection.

    The registry scans one or more directories for ``SKILL.md`` files,
    parses them, and stores the resulting :class:`Skill` objects for
    later retrieval by name, tag, or free-text search.

    Attributes:
        _skills: Internal mapping of skill name to :class:`Skill` instance.

    Example:
        >>> registry = SkillRegistry()
        >>> registry.discover("./skills")
        >>> skill = registry.get("web_research")
        >>> print(skill.to_prompt_section())
    """

    def __init__(self) -> None:
        """Initialize the SkillRegistry with an empty internal skill store."""
        self._skills: dict[str, Skill] = {}

    @property
    def skill_names(self) -> list[str]:
        """Return the names of all registered skills.

        Returns:
            A list of skill name strings in insertion order.
        """
        return list(self._skills.keys())

    def discover(self, *directories: str | Path) -> list[str]:
        """Recursively scan directories for SKILL.md files and register them.

        Each directory is walked recursively.  When a ``SKILL.md`` file is
        found it is scanned for prompt-injection threats, parsed, and
        registered under its metadata name.  Duplicate skill names are
        skipped (first-discovered wins).

        Args:
            *directories: One or more directory paths to scan.  Non-existent
                directories are logged as warnings and skipped.

        Returns:
            A list of skill names that were newly registered during this
            discovery pass.
        """
        from xerxes.security.prompt_scanner import scan_context_content

        discovered = []
        for directory in directories:
            dir_path = Path(directory)
            if not dir_path.is_dir():
                logger.warning("Skill directory not found: %s", dir_path)
                continue

            for skill_file in dir_path.rglob("SKILL.md"):
                try:
                    content = skill_file.read_text(encoding="utf-8")
                    safe = scan_context_content(content, filename=f"SKILL.md: {skill_file}")
                    if safe.startswith("[BLOCKED:"):
                        logger.warning("Blocked skill file %s due to security scan", skill_file)
                        continue
                    skill = parse_skill_md(safe, skill_file)
                    if skill.name not in self._skills:
                        self._skills[skill.name] = skill
                        discovered.append(skill.name)
                        logger.info("Discovered skill: %s at %s", skill.name, skill_file)
                    else:
                        logger.debug("Skill %s already registered, skipping %s", skill.name, skill_file)
                except Exception:
                    logger.warning("Failed to parse skill at %s", skill_file, exc_info=True)

        return discovered

    def register(self, skill: Skill) -> None:
        """Manually register a pre-built :class:`Skill` instance.

        Overwrites any existing skill with the same name.

        Args:
            skill: The :class:`Skill` to register.
        """
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        """Look up a registered skill by name.

        Args:
            name: The skill name to look up.

        Returns:
            The :class:`Skill` instance, or ``None`` if not found.
        """
        return self._skills.get(name)

    def get_all(self) -> list[Skill]:
        """Return all registered skills.

        Returns:
            A list of all :class:`Skill` instances currently registered.
        """
        return list(self._skills.values())

    def search(self, query: str = "", tags: list[str] | None = None) -> list[Skill]:
        """Search skills by free-text query and/or tag matching.

        When *query* is provided, skills whose name or description contains
        the query (case-insensitive) are included.  When *tags* is provided,
        skills that have at least one matching tag are included.  If neither
        *query* nor *tags* is provided, all skills are returned.

        Args:
            query: A case-insensitive substring to search for in skill names
                and descriptions.  Defaults to ``""``.
            tags: An optional list of tags to filter by.  A skill matches if
                it has any of the specified tags.

        Returns:
            A list of matching :class:`Skill` instances.
        """
        results = []
        query_lower = query.lower()
        for skill in self._skills.values():
            if query_lower and (query_lower in skill.name.lower() or query_lower in skill.metadata.description.lower()):
                results.append(skill)
            elif tags and any(tag in skill.metadata.tags for tag in tags):
                results.append(skill)
            elif not query and not tags:
                results.append(skill)
        return results

    def validate_dependencies(self, plugin_registry: tp.Any = None) -> list[str]:
        """Validate that all registered skills have their dependencies met.

        Args:
            plugin_registry: Optional PluginRegistry instance to check
                required_tools against.

        Returns:
            List of error messages (empty if all dependencies are satisfied).
        """
        errors: list[str] = []
        for name, skill in self._skills.items():
            for dep in skill.metadata.dependencies:
                if dep not in self._skills:
                    errors.append(f"Skill '{name}' requires missing dependency '{dep}'")
            if plugin_registry is not None:
                for tool_name in skill.metadata.required_tools:
                    if plugin_registry.get_tool(tool_name) is None:
                        errors.append(f"Skill '{name}' requires missing tool '{tool_name}'")
        return errors

    def build_skills_index(self) -> str:
        """Build a compact plain-text index of all registered skills.

        The index lists each skill's name, description, and tags on a
        single indented line, suitable for injection into a system prompt
        so the agent knows which skills are available.

        Returns:
            A multi-line string listing available skills, or an empty string
            if no skills are registered.
        """
        if not self._skills:
            return ""
        lines = ["Available skills:"]
        for skill in self._skills.values():
            desc = skill.metadata.description or "No description"
            tags = ", ".join(skill.metadata.tags) if skill.metadata.tags else ""
            tag_str = f" [{tags}]" if tags else ""
            lines.append(f"  - {skill.name}: {desc}{tag_str}")
        return "\n".join(lines)


def skill_matches_platform(skill: Skill, current_platform: str | None = None) -> bool:
    """Return True when *skill* is compatible with the current OS.

    Skills declare platform requirements via a ``platforms`` list in their
    YAML frontmatter.  If the field is absent or empty the skill is
    compatible with **all** platforms.

    Args:
        skill: The skill to check.
        current_platform: Override platform string (defaults to ``sys.platform``).

    Returns:
        True if the skill has no platform restrictions or matches the
        current platform.
    """
    platforms = skill.metadata.platforms
    if not platforms:
        return True
    current = (current_platform or sys.platform).lower()
    for platform in platforms:
        normalized = str(platform).lower().strip()
        mapped = _PLATFORM_MAP.get(normalized, normalized)
        if current.startswith(mapped):
            return True
    return False


def _load_skill_config() -> dict[str, dict[str, tp.Any]]:
    """Read ``skills.config`` from ``~/.xerxes/config.yaml``.

    Returns:
        A dict mapping ``skill_name → {key: value}`` of configured values,
        or an empty dict if the file is missing or unparseable.
    """
    from xerxes.core.paths import xerxes_subdir

    config_path = xerxes_subdir("config.yaml")
    if not config_path.exists():
        return {}
    try:
        import yaml

        parsed = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    skills_cfg = parsed.get("skills")
    if not isinstance(skills_cfg, dict):
        return {}
    config_section = skills_cfg.get("config")
    if not isinstance(config_section, dict):
        return {}
    return config_section


def resolve_skill_config(
    skill: Skill,
    user_config: dict[str, dict[str, tp.Any]] | None = None,
) -> dict[str, tp.Any]:
    """Resolve config values for a skill from user config.

    Args:
        skill: The skill whose ``config_vars`` to resolve.
        user_config: Pre-loaded user config mapping.  If *None*, loaded
            from disk automatically.

    Returns:
        A dict of ``var_name → value`` for every ``config_var`` declared
        by the skill.  Missing values are omitted.
    """
    if user_config is None:
        user_config = _load_skill_config()
    skill_cfg = user_config.get(skill.name, {})
    result: dict[str, tp.Any] = {}
    for var in skill.metadata.config_vars:
        if var in skill_cfg:
            result[var] = skill_cfg[var]
    return result


def inject_skill_config(skill: Skill, user_config: dict[str, dict[str, tp.Any]] | None = None) -> str:
    """Build a config-injection block for a skill message.

    Args:
        skill: The skill to build the block for.
        user_config: Pre-loaded user config mapping.  If *None*, loaded
            from disk automatically.

    Returns:
        A formatted config block string, or an empty string if the skill
        declares no config vars or none are set.
    """
    resolved = resolve_skill_config(skill, user_config)
    if not resolved:
        return ""
    lines = ["", "[Skill config (from ~/.xerxes/config.yaml):"]
    for key, value in sorted(resolved.items()):
        display_val = str(value) if value is not None else "(not set)"
        lines.append(f"  {key} = {display_val}")
    lines.append("]")
    return "\n".join(lines)
