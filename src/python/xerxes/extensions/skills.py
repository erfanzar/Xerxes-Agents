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
    """Mark ``name`` as active in the module-global skill set."""

    global _active_skills
    if name not in _active_skills:
        _active_skills.append(name)


def get_active_skills() -> list[str]:
    """Return a snapshot of currently active skill names."""

    return list(_active_skills)


def clear_active_skills() -> None:
    """Reset the module-global active skill set."""

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
        name: Skill identifier and registry key.
        description: Human-readable summary used in listings.
        version: Semver string.
        tags: Categorisation labels used for search.
        resources: Resource file names that determine ``resources_dir``.
        author: Author name.
        dependencies: Skill names this skill depends on.
        required_tools: Tool names this skill requires.
        platforms: Supported platforms; consulted by ``skill_matches_platform``.
        config_vars: Configurable variable names consumed by ``resolve_skill_config``.
        trust_level: Trust tier (e.g. ``community``).
        source: Origin label (e.g. ``local``).
        setup_command: Shell command run once during setup.
        subcommands: Sub-command names exposed by the skill (e.g. for slash
            command completion).
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
    subcommands: list[str] = field(default_factory=list)


@dataclass
class Skill:
    """Complete skill object combining metadata, instructions, and paths.

    Attributes:
        metadata: Parsed frontmatter.
        instructions: Markdown body after the frontmatter, used for prompt injection.
        source_path: Path to the originating ``SKILL.md`` file.
        resources_dir: Directory containing companion resources, when present.
    """

    metadata: SkillMetadata
    instructions: str
    source_path: Path
    resources_dir: Path | None = None

    @property
    def name(self) -> str:
        """Return ``self.metadata.name``."""

        return self.metadata.name

    def to_prompt_section(self) -> str:
        """Render the skill as a markdown prompt section.

        Returns:
            ``## Skill: {name}`` header followed by the description and
            instruction body.
        """

        header = f"## Skill: {self.metadata.name}"
        if self.metadata.description:
            header += f"\n{self.metadata.description}"
        return f"{header}\n\n{self.instructions}"


def parse_skill_md(content: str, source_path: Path) -> Skill:
    """Parse a ``SKILL.md`` document into a ``Skill``.

    Supports YAML frontmatter delimited by ``---``. Falls back to simple
    ``key: value`` parsing when PyYAML is unavailable.

    Args:
        content: Full markdown text including optional frontmatter.
        source_path: Path to the originating file; recorded on the result.

    Returns:
        Populated ``Skill`` instance.
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
        """Coerce a raw frontmatter value into a clean ``list[str]``."""

        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()]
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return []

    # Sub-commands: explicit ``subcommands:`` field wins; otherwise auto-detect
    # from sibling ``references/<sub>-workflow.md`` files (the autoresearch
    # skill's convention). This means existing community skills get correct
    # sub-command completion without any frontmatter changes.
    explicit_subcommands = _normalize_list(metadata_dict.get("subcommands"))
    detected_subcommands = _detect_subcommands_from_references(source_path)
    subcommands = explicit_subcommands or detected_subcommands

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
        subcommands=subcommands,
    )

    resources_dir = source_path.parent if metadata.resources else None

    return Skill(
        metadata=metadata,
        instructions=body,
        source_path=source_path,
        resources_dir=resources_dir,
    )


def _detect_subcommands_from_references(source_path: Path) -> list[str]:
    """Infer sub-commands from sibling ``references/<sub>-workflow.md`` files.

    Used as a fallback when the skill author did not declare ``subcommands:``
    in frontmatter. Mirrors the autoresearch convention: each workflow file
    becomes a sub-command (``autoresearch:fix`` <- ``references/fix-workflow.md``).
    """

    references_dir = source_path.parent / "references"
    if not references_dir.is_dir():
        return []
    subs: list[str] = []
    for path in sorted(references_dir.glob("*-workflow.md")):
        sub = path.stem[: -len("-workflow")]
        if sub:
            subs.append(sub)
    return subs


class SkillRegistry:
    """In-memory registry of discovered skills."""

    def __init__(self) -> None:
        """Initialize an empty skill index."""

        self._skills: dict[str, Skill] = {}

    @property
    def skill_names(self) -> list[str]:
        """Return a snapshot of registered skill names."""

        return list(self._skills.keys())

    def discover(self, *directories: str | Path) -> list[str]:
        """Recursively scan ``directories`` for ``SKILL.md`` files.

        Each file is content-scanned for prompt injection before being parsed.
        Existing entries are not overwritten.

        Returns:
            Names of newly registered skills.
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
        """Store ``skill`` in the registry, overwriting any prior entry."""

        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        """Return the skill registered under ``name``, or ``None``."""

        return self._skills.get(name)

    def get_all(self) -> list[Skill]:
        """Return a snapshot of every registered skill."""

        return list(self._skills.values())

    def search(self, query: str = "", tags: list[str] | None = None) -> list[Skill]:
        """Filter skills by case-insensitive ``query`` text or ``tags`` membership.

        With no filters supplied the full skill list is returned.
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
        """Verify that each skill's dependencies and required tools exist.

        Args:
            plugin_registry: Optional plugin registry exposing ``get_tool``;
                when present, ``required_tools`` are validated as well.

        Returns:
            Human-readable error strings for unmet requirements.
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
        """Return a markdown bulleted list of registered skills, or ``""``."""

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
    """Return whether ``skill`` is compatible with ``current_platform``.

    A skill with no declared platforms is treated as universal.

    Args:
        skill: Skill to test.
        current_platform: Override platform string; defaults to ``sys.platform``.
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
    """Return the ``skills.config`` block of ``~/.xerxes/config.yaml``.

    Yields an empty dict when the file is missing or unreadable.
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
    """Return the user config subset that matches ``skill.metadata.config_vars``.

    Args:
        skill: Skill whose declared config variables drive the extraction.
        user_config: Optional override; defaults to ``_load_skill_config()``.
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
    """Render resolved skill config as a markdown snippet for prompt injection.

    Returns the empty string when the skill defines no resolved variables.
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
