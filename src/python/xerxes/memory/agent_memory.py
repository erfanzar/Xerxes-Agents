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
"""Agent self-knowledge and continuous learning system.

Every agent has a persistent read-write memory under ``~/.xerxes/agent_memory/``:

- ``user_taste.md`` — What the user likes/dislikes, communication style,
  preferred tools, common workflows, frustrations.
- ``project_context.md`` — AGENTS.md, XERXES.md, and project conventions
  the agent has learned.
- ``skill_journal.md`` — Patterns observed that could become skills.
- ``self_reflection.md`` — Agent's own notes about what worked/didn't.

The agent reads these at session start and updates them after each turn.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..core.paths import xerxes_subdir

logger = logging.getLogger(__name__)

# Files in the agent's read-write memory
_AGENT_MEMORY_FILES = {
    "user_taste": "user_taste.md",
    "project_context": "project_context.md",
    "skill_journal": "skill_journal.md",
    "self_reflection": "self_reflection.md",
    "tool_usage_patterns": "tool_usage_patterns.md",
}


class AgentMemory:
    """Persistent read-write memory for an agent.

    Each agent (identified by agent_id) gets its own directory under
    ``~/.xerxes/agent_memory/<agent_id>/``. The agent can read/write
    these files at any time.
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._dir = xerxes_subdir("agent_memory", agent_id)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        """Create default memory files if they don't exist."""
        for key, filename in _AGENT_MEMORY_FILES.items():
            path = self._dir / filename
            if not path.exists():
                path.write_text(self._default_content(key))

    def _default_content(self, key: str) -> str:
        """Return default content for a memory file."""
        defaults = {
            "user_taste": "# User Taste Profile\n\n## Communication Style\n-\n\n## Preferred Tools\n-\n\n## Common Workflows\n-\n\n## Frustrations / Avoid\n-\n\n## Notes\n\n",
            "project_context": "# Project Context\n\n## AGENTS.md Summary\n\n## XERXES.md Summary\n\n## Project Conventions\n-\n\n## Important Files\n-\n\n",
            "skill_journal": "# Skill Journal\n\n## Observed Patterns\n\n## Proposed Skills\n\n## Implemented Skills\n\n",
            "self_reflection": "# Self Reflection\n\n## What Worked\n\n## What Didn't\n\n## Improvements\n\n",
            "tool_usage_patterns": "# Tool Usage Patterns\n\n## Frequently Used Tools\n\n## Tool Combinations\n\n## Success Patterns\n\n",
        }
        return defaults.get(key, "")

    # ── Read ──────────────────────────────────────────────────────

    def read(self, key: str) -> str:
        """Read a memory file by key."""
        filename = _AGENT_MEMORY_FILES.get(key)
        if not filename:
            return ""
        path = self._dir / filename
        if path.exists():
            return path.read_text()
        return ""

    def read_all(self) -> dict[str, str]:
        """Read all memory files."""
        return {key: self.read(key) for key in _AGENT_MEMORY_FILES}

    # ── Write ─────────────────────────────────────────────────────

    def write(self, key: str, content: str) -> None:
        """Overwrite a memory file."""
        filename = _AGENT_MEMORY_FILES.get(key)
        if not filename:
            return
        path = self._dir / filename
        path.write_text(content)
        logger.debug("Agent %s wrote %s (%d chars)", self.agent_id, key, len(content))

    def append(self, key: str, content: str) -> None:
        """Append to a memory file."""
        filename = _AGENT_MEMORY_FILES.get(key)
        if not filename:
            return
        path = self._dir / filename
        existing = path.read_text() if path.exists() else ""
        path.write_text(existing + "\n" + content)

    def patch(self, key: str, old_text: str, new_text: str) -> bool:
        """Patch a memory file (find-and-replace)."""
        filename = _AGENT_MEMORY_FILES.get(key)
        if not filename:
            return False
        path = self._dir / filename
        content = path.read_text()
        if old_text not in content:
            return False
        path.write_text(content.replace(old_text, new_text, 1))
        return True

    # ── Project Context Integration ───────────────────────────────

    def sync_project_context(self, cwd: Path | None = None) -> None:
        """Read AGENTS.md, XERXES.md, USER.md from project and update memory."""
        if cwd is None:
            cwd = Path.cwd()

        # Look for project files
        agents_md = self._read_project_file(cwd, "AGENTS.md")
        xerxes_md = self._read_project_file(cwd, "XERXES.md")
        user_md = self._read_project_file(cwd, "USER.md")
        soul_md = self._read_project_file(cwd, "SOUL.md")

        # Update project_context.md
        context_parts = ["# Project Context\n"]
        if agents_md:
            context_parts.append(f"\n## AGENTS.md\n```\n{agents_md[:2000]}\n```\n")
        if xerxes_md:
            context_parts.append(f"\n## XERXES.md\n```\n{xerxes_md[:2000]}\n```\n")
        if user_md:
            context_parts.append(f"\n## USER.md\n```\n{user_md[:2000]}\n```\n")
        if soul_md:
            context_parts.append(f"\n## SOUL.md\n```\n{soul_md[:2000]}\n```\n")

        self.write("project_context", "\n".join(context_parts))
        logger.info("Agent %s synced project context from %s", self.agent_id, cwd)

    def _read_project_file(self, cwd: Path, filename: str) -> str:
        """Read a file from cwd or any parent directory."""
        for p in [cwd, *list(cwd.parents)]:
            candidate = p / filename
            if candidate.exists():
                try:
                    return candidate.read_text()
                except Exception:
                    continue
        return ""

    # ── Learning from Interactions ────────────────────────────────

    def learn_from_interaction(
        self,
        user_message: str,
        agent_response: str,
        tools_used: list[str],
        success: bool,
    ) -> None:
        """Extract learnings from an interaction and update memory."""
        # Update tool usage patterns
        tool_entry = f"- {', '.join(tools_used)} → {'✓' if success else '✗'}"
        self.append("tool_usage_patterns", tool_entry)

        # Update self-reflection
        reflection = f"\n## Turn at {self._now()}\n"
        reflection += f"User: {user_message[:200]}\n"
        reflection += f"Tools: {tools_used}\n"
        reflection += f"Result: {'success' if success else 'failure'}\n"
        self.append("self_reflection", reflection)

    def update_user_taste(
        self,
        preference: str,
        category: str = "notes",
    ) -> None:
        """Add a user preference observation."""
        entry = f"\n- {preference}"
        # Try to append under the right section
        content = self.read("user_taste")
        section_header = f"## {category.replace('_', ' ').title()}"
        if section_header in content:
            # Insert after the section header
            parts = content.split(section_header, 1)
            if len(parts) == 2:
                after_header = parts[1]
                # Find next ## or end
                next_section = after_header.find("\n## ")
                if next_section == -1:
                    new_content = parts[0] + section_header + after_header + entry + "\n"
                else:
                    new_content = (
                        parts[0]
                        + section_header
                        + after_header[:next_section]
                        + entry
                        + "\n"
                        + after_header[next_section:]
                    )
                self.write("user_taste", new_content)
                return
        # Fallback: just append
        self.append("user_taste", entry)

    def propose_skill(self, name: str, description: str, pattern: str) -> None:
        """Propose a new skill based on observed pattern."""
        entry = f"\n### {name}\n- Description: {description}\n- Pattern: {pattern}\n- Status: proposed\n"
        self.append("skill_journal", entry)

    def mark_skill_implemented(self, name: str) -> None:
        """Mark a proposed skill as implemented."""
        content = self.read("skill_journal")
        old = f"### {name}\n- Description:"
        if old in content:
            # Find and update status
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if f"### {name}" in line:
                    # Find the status line and update it
                    for j in range(i, min(i + 10, len(lines))):
                        if "Status: proposed" in lines[j]:
                            lines[j] = lines[j].replace("proposed", "implemented")
                            break
                    break
            self.write("skill_journal", "\n".join(lines))

    # ── System Prompt Integration ─────────────────────────────────

    def get_system_prompt_addendum(self) -> str:
        """Return memory content formatted for injection into system prompt."""
        parts = []
        user_taste = self.read("user_taste")
        if user_taste.strip():
            parts.append(f"[User Taste Profile]\n{user_taste}")

        project_context = self.read("project_context")
        if project_context.strip():
            parts.append(f"[Project Context]\n{project_context}")

        tool_patterns = self.read("tool_usage_patterns")
        if tool_patterns.strip():
            parts.append(f"[Tool Usage Patterns]\n{tool_patterns}")

        result = "\n\n".join(parts)
        if result:
            return (
                "MEMORY INSTRUCTION: You have persistent memory. ALWAYS read from memory at the start "
                "of each session and ALWAYS write important observations to memory.\n\n"
                + result
            )
        return ""

    def _now(self) -> str:
        from datetime import UTC, datetime
        return datetime.now(UTC).isoformat()


# Global registry of agent memories
_agent_memories: dict[str, AgentMemory] = {}


def get_agent_memory(agent_id: str) -> AgentMemory:
    """Get or create an AgentMemory for the given agent_id."""
    if agent_id not in _agent_memories:
        _agent_memories[agent_id] = AgentMemory(agent_id)
    return _agent_memories[agent_id]


def list_agent_memories() -> list[str]:
    """List all agent IDs that have memory."""
    base = xerxes_subdir("agent_memory")
    if not base.exists():
        return []
    return [d.name for d in base.iterdir() if d.is_dir()]
