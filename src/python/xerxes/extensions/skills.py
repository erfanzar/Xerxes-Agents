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
"""Skill definition, parsing, registry, and platform matching.

Skills are reusable instruction sets stored as ``SKILL.md`` files with YAML
frontmatter. ``SkillRegistry`` discovers them on disk, and helpers like
``skill_matches_platform`` filter by OS compatibility.
"""

from __future__ import annotations

import logging
import re
import sys
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_active_skills: list[str] = []


def activate_skill(name: str) -> None:
    """Add a skill name to the global active set.

    Args:
        name (str): IN: Skill identifier. OUT: Appended to ``_active_skills``
            if not already present.

    Returns:
        None: OUT: Global state is updated.
    """

    global _active_skills
    if name not in _active_skills:
        _active_skills.append(name)


def get_active_skills() -> list[str]:
    """Return the current global active skill names.

    Returns:
        list[str]: OUT: Snapshot of ``_active_skills``.
    """

    return list(_active_skills)


def clear_active_skills() -> None:
    """Remove all entries from the global active skill set.

    Returns:
        None: OUT: ``_active_skills`` is emptied.
    """

    global _active_skills
    _active_skills.clear()


_PLATFORM_MAP = {
    "macos": "darwin",
    "linux": "linux",
    "windows": "win32",
}


@dataclass
class SkillMetadata:
    """Parsed frontmatter fields for a skill.

    Attributes:
        name (str): IN: Skill identifier. OUT: Registry key.
        description (str): IN: Human summary. OUT: Used in listings.
        version (str): IN: Semver string. OUT: Stored.
        tags (list[str]): IN: Categorisation labels. OUT: Used for search.
        resources (list[str]): IN: Resource file names. OUT: Used to set
            ``resources_dir``.
        author (str): IN: Author name. OUT: Stored.
        dependencies (list[str]): IN: Skill dependency names. OUT: Validated
            by registry.
        required_tools (list[str]): IN: Tool names needed. OUT: Validated by
            registry.
        platforms (list[str]): IN: Supported platform names. OUT: Matched by
            ``skill_matches_platform``.
        config_vars (list[str]): IN: Configurable variable names. OUT: Used
            by ``resolve_skill_config``.
        trust_level (str): IN: Trust tier. OUT: Stored.
        source (str): IN: Origin label. OUT: Stored.
        setup_command (str): IN: Shell command for one-time setup. OUT:
            Stored.
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
    """Complete skill object combining metadata, instructions, and paths.

    Attributes:
        metadata (SkillMetadata): IN: Parsed frontmatter. OUT: Stored.
        instructions (str): IN: Markdown body after frontmatter. OUT: Used
            for prompt injection.
        source_path (Path): IN: Path to the ``SKILL.md`` file. OUT: Stored.
        resources_dir (Path | None): IN: Directory containing resources.
            OUT: Set when ``metadata.resources`` is non-empty.
    """

    metadata: SkillMetadata
    instructions: str
    source_path: Path
    resources_dir: Path | None = None

    @property
    def name(self) -> str:
        """Convenience accessor for ``metadata.name``.

        Returns:
            str: OUT: The skill name.
        """

        return self.metadata.name

    def to_prompt_section(self) -> str:
        """Render the skill as a markdown prompt section.

        Returns:
            str: OUT: ``## Skill: {name}`` header plus description and
            instructions.
        """

        header = f"## Skill: {self.metadata.name}"
        if self.metadata.description:
            header += f"\n{self.metadata.description}"
        return f"{header}\n\n{self.instructions}"


def parse_skill_md(content: str, source_path: Path) -> Skill:
    """Parse a ``SKILL.md`` string into a ``Skill`` object.

    Supports YAML frontmatter delimited by ``---``. Falls back to simple
    ``key: value`` parsing if PyYAML is unavailable.

    Args:
        content (str): IN: Full markdown text. OUT: Split into frontmatter
            and body.
        source_path (Path): IN: Path to the file. OUT: Stored on the
            returned ``Skill``.

    Returns:
        Skill: OUT: Populated skill instance.
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
        """Coerce a frontmatter value into a list of strings.

        Args:
            value: IN: Raw parsed value. OUT: Normalised to ``list[str]``.

        Returns:
            list[str]: OUT: Clean string list.
        """

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
    """In-memory registry of discovered skills."""

    def __init__(self) -> None:
        """Initialize an empty skill index."""

        self._skills: dict[str, Skill] = {}

    @property
    def skill_names(self) -> list[str]:
        """Return all registered skill names.

        Returns:
            list[str]: OUT: Snapshot of ``self._skills`` keys.
        """

        return list(self._skills.keys())

    def discover(self, *directories: str | Path) -> list[str]:
        """Scan directories recursively for ``SKILL.md`` files.

        Args:
            *directories (str | Path): IN: Directories to scan. OUT:
                Recursively searched.

        Returns:
            list[str]: OUT: Names of newly discovered skills.
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
        """Manually add a skill to the registry.

        Args:
            skill (Skill): IN: Skill instance. OUT: Stored by ``skill.name``.

        Returns:
            None: OUT: Overwrites any existing skill with the same name.
        """

        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        """Retrieve a skill by name.

        Args:
            name (str): IN: Skill identifier. OUT: Looked up in the registry.

        Returns:
            Skill | None: OUT: Skill instance or ``None``.
        """

        return self._skills.get(name)

    def get_all(self) -> list[Skill]:
        """Return all registered skills.

        Returns:
            list[Skill]: OUT: Snapshot of stored skills.
        """

        return list(self._skills.values())

    def search(self, query: str = "", tags: list[str] | None = None) -> list[Skill]:
        """Filter skills by query text or tag membership.

        Args:
            query (str): IN: Substring to match against name/description.
                OUT: Case-insensitive filter.
            tags (list[str] | None): IN: Tags to match. OUT: Skills with any
                matching tag are included.

        Returns:
            list[Skill]: OUT: Matching skills; if no filters given, returns
            all skills.
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
        """Check that every skill's dependencies and required tools exist.

        Args:
            plugin_registry (tp.Any): IN: Optional plugin registry with a
                ``get_tool`` method. OUT: Used to validate
                ``required_tools``.

        Returns:
            list[str]: OUT: Human-readable error strings.
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
        """Build a markdown list of available skills.

        Returns:
            str: OUT: Formatted index or empty string if no skills.
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
    """Determine whether a skill supports the current OS.

    Args:
        skill (Skill): IN: Skill to test. OUT: ``metadata.platforms`` is
            checked.
        current_platform (str | None): IN: Override platform string (defaults
            to ``sys.platform``). OUT: Normalised and compared.

    Returns:
        bool: OUT: ``True`` if no platforms are specified or if the current
        platform matches.
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
    """Load per-skill configuration from ``~/.xerxes/config.yaml``.

    Returns:
        dict[str, dict[str, tp.Any]]: OUT: Mapping from skill name to config
        dict; empty if file missing or unreadable.
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
    """Resolve configuration values for a skill from user config.

    Args:
        skill (Skill): IN: Skill whose ``config_vars`` are requested. OUT:
            Used to determine which keys to extract.
        user_config (dict[str, dict[str, tp.Any]] | None): IN: Override
            config mapping. OUT: Defaults to ``_load_skill_config()``.

    Returns:
        dict[str, tp.Any]: OUT: Subset of config containing only the skill's
        declared variables.
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
    """Format resolved skill configuration as a markdown snippet.

    Args:
        skill (Skill): IN: Skill to resolve config for. OUT: Passed to
            ``resolve_skill_config``.
        user_config (dict[str, dict[str, tp.Any]] | None): IN: Override
            config mapping. OUT: Passed to ``resolve_skill_config``.

    Returns:
        str: OUT: Markdown block with key/value pairs, or empty string if no
        config variables are set.
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
