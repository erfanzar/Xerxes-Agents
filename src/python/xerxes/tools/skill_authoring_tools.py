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
"""Skill authoring tools for agents to write reusable skills.

Skills are written as SKILL.md files with YAML frontmatter and markdown body.
They are saved to ~/.xerxes/skills/ so the SkillRegistry can discover and load them.

Example SKILL.md format:
---
name: fibonacci-generator
description: Generate Fibonacci sequences efficiently
version: 1.0.0
tags: [math, python, algorithm]
required_tools: [ExecuteShell, WriteFile]
---

# When to use
Apply this skill when the user asks for Fibonacci numbers, sequences, or related mathematical patterns.

# How to use
1. Use ExecuteShell to calculate the sequence
2. Use WriteFile to save results if requested

# Example
```python
def fibonacci(n):
    a, b = 0, 1
    result = []
    for _ in range(n):
        result.append(a)
        a, b = b, a + b
    return result
```
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from ..core.paths import xerxes_subdir
from ..extensions.skills import parse_skill_md

logger = logging.getLogger(__name__)


def _slugify(text: str, max_len: int = 40) -> str:
    """Lowercase text, replace non-alphanumerics with hyphens, and truncate."""
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    if not text:
        text = f"skill-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return text[:max_len].rstrip("-") or "skill"


def _validate_skill_md(content: str) -> tuple[bool, str]:
    """Validate a SKILL.md document. Returns (is_valid, error_message)."""
    # Check for YAML frontmatter
    if not content.startswith("---"):
        return False, "SKILL.md must start with YAML frontmatter (---)"

    # Check for required fields
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", content, re.DOTALL)
    if not fm_match:
        return False, "Invalid frontmatter format"

    fm_text = fm_match.group(1)

    # Parse frontmatter
    metadata: dict[str, Any] = {}
    try:
        import yaml
        metadata = yaml.safe_load(fm_text) or {}
    except ImportError:
        # Fallback parsing
        for line in fm_text.strip().splitlines():
            line = line.strip()
            if ":" in line:
                key, _, value = line.partition(":")
                metadata[key.strip()] = value.strip().strip('"').strip("'")

    # Check required fields
    if "name" not in metadata:
        return False, "Frontmatter must include 'name' field"
    if "description" not in metadata:
        return False, "Frontmatter must include 'description' field"
    if "version" not in metadata:
        return False, "Frontmatter must include 'version' field"

    # Check body content
    body = fm_match.group(2).strip()
    if not body:
        return False, "SKILL.md must have a markdown body after frontmatter"

    if "# When to use" not in body and "# How to use" not in body:
        return False, "SKILL.md body should include '# When to use' or '# How to use' sections"

    return True, ""


class skill_write:
    """Write a new SKILL.md file to the agent's skills directory.

    The skill will be saved to ~/.xerxes/skills/<skill_name>/SKILL.md
    and will be discoverable by the SkillRegistry.
    """

    @staticmethod
    def check_fn() -> bool:
        return True

    @staticmethod
    def get_schema() -> dict[str, Any]:
        return {
            "name": "skill_write",
            "description": "Write a new skill as a SKILL.md file with YAML frontmatter. The skill will be saved to ~/.xerxes/skills/ and loaded by the SkillRegistry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill identifier (e.g., 'fibonacci-generator'). Will be slugified.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Short description of what the skill does.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full SKILL.md content including YAML frontmatter (---) and markdown body. Must include name, description, version in frontmatter and # When to use / # How to use in body.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization (e.g., ['math', 'python']).",
                        "default": [],
                    },
                    "required_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tools this skill requires (e.g., ['ExecuteShell', 'WriteFile']).",
                        "default": [],
                    },
                    "version": {
                        "type": "string",
                        "description": "Semver version string.",
                        "default": "1.0.0",
                    },
                },
                "required": ["name", "description", "content"],
            },
        }

    @staticmethod
    def static_call(
        name: str,
        description: str,
        content: str,
        tags: list[str] | None = None,
        required_tools: list[str] | None = None,
        version: str = "1.0.0",
    ) -> str:
        # Validate the content
        is_valid, _error = _validate_skill_md(content)
        if not is_valid:
            # Auto-fix: generate proper frontmatter if missing
            content = _generate_skill_md(name, description, content, tags or [], required_tools or [], version)

        # Slugify the name for directory
        skill_dir_name = _slugify(name)
        skills_dir = xerxes_subdir("skills")
        skill_dir = skills_dir / skill_dir_name
        skill_dir.mkdir(parents=True, exist_ok=True)

        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(content)

        # Try to parse and validate
        try:
            skill = parse_skill_md(content, skill_path)
            logger.info("Skill '%s' written to %s", skill.name, skill_path)
            return f"Skill '{skill.name}' written to {skill_path}. Tags: {skill.metadata.tags}. Tools: {skill.metadata.required_tools}"
        except Exception as e:
            logger.warning("Skill written but parse failed: %s", e)
            return f"Skill written to {skill_path} but validation failed: {e}"


class skill_read:
    """Read a skill's SKILL.md content."""

    @staticmethod
    def check_fn() -> bool:
        return True

    @staticmethod
    def get_schema() -> dict[str, Any]:
        return {
            "name": "skill_read",
            "description": "Read a skill's SKILL.md content from ~/.xerxes/skills/.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name (directory name or slug).",
                    },
                },
                "required": ["name"],
            },
        }

    @staticmethod
    def static_call(name: str) -> str:
        skills_dir = xerxes_subdir("skills")
        skill_dir = skills_dir / _slugify(name)
        skill_path = skill_dir / "SKILL.md"

        if not skill_path.exists():
            # Try searching
            for subdir in skills_dir.iterdir():
                if subdir.is_dir() and subdir.name == _slugify(name):
                    skill_path = subdir / "SKILL.md"
                    break

        if not skill_path.exists():
            return f"Skill '{name}' not found in {skills_dir}"

        return skill_path.read_text()


class skill_list:
    """List all skills in the agent's skills directory."""

    @staticmethod
    def check_fn() -> bool:
        return True

    @staticmethod
    def get_schema() -> dict[str, Any]:
        return {
            "name": "skill_list",
            "description": "List all skills in ~/.xerxes/skills/ with their metadata.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }

    @staticmethod
    def static_call() -> str:
        skills_dir = xerxes_subdir("skills")
        if not skills_dir.exists():
            return "No skills directory found."

        skills = []
        for subdir in sorted(skills_dir.iterdir()):
            if not subdir.is_dir():
                continue
            skill_md = subdir / "SKILL.md"
            if not skill_md.exists():
                continue

            try:
                content = skill_md.read_text()
                skill = parse_skill_md(content, skill_md)
                skills.append({
                    "name": skill.name,
                    "description": skill.metadata.description,
                    "version": skill.metadata.version,
                    "tags": skill.metadata.tags,
                    "path": str(skill_md),
                })
            except Exception as e:
                skills.append({
                    "name": subdir.name,
                    "description": f"Parse error: {e}",
                    "version": "?",
                    "tags": [],
                    "path": str(skill_md),
                })

        if not skills:
            return "No skills found."

        import json
        return json.dumps(skills, indent=2, ensure_ascii=False)


class skill_update:
    """Update an existing skill's SKILL.md content."""

    @staticmethod
    def check_fn() -> bool:
        return True

    @staticmethod
    def get_schema() -> dict[str, Any]:
        return {
            "name": "skill_update",
            "description": "Update an existing skill by patching its SKILL.md content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name to update.",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Text to find and replace.",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                },
                "required": ["name", "old_text", "new_text"],
            },
        }

    @staticmethod
    def static_call(name: str, old_text: str, new_text: str) -> str:
        skills_dir = xerxes_subdir("skills")
        skill_dir = skills_dir / _slugify(name)
        skill_path = skill_dir / "SKILL.md"

        if not skill_path.exists():
            return f"Skill '{name}' not found."

        content = skill_path.read_text()
        if old_text not in content:
            return f"Text not found in skill '{name}'."

        content = content.replace(old_text, new_text, 1)
        skill_path.write_text(content)

        return f"Skill '{name}' updated."


def _generate_skill_md(
    name: str,
    description: str,
    body: str,
    tags: list[str],
    required_tools: list[str],
    version: str,
) -> str:
    """Generate a properly formatted SKILL.md from partial content."""
    # Extract body without any existing frontmatter
    body_clean = body
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", body, re.DOTALL)
    if fm_match:
        body_clean = fm_match.group(2).strip()

    # Ensure required sections
    if "# When to use" not in body_clean:
        body_clean = "# When to use\n\n" + body_clean

    if "# How to use" not in body_clean:
        body_clean += "\n\n# How to use\n\n1. Analyze the user's request\n2. Apply the appropriate tools\n3. Verify the result\n"

    # Build frontmatter
    fm_lines = [
        "---",
        f"name: {_slugify(name)}",
        f'description: "{description}"',
        f"version: {version}",
    ]
    if tags:
        fm_lines.append(f"tags: [{', '.join(tags)}]")
    if required_tools:
        fm_lines.append(f"required_tools: [{', '.join(required_tools)}]")
    fm_lines.append("---")

    return "\n".join(fm_lines) + "\n\n" + body_clean
