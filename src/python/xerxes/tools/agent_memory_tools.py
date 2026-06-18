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
"""Agent self-memory tools — read/write agent's own persistent knowledge.

These tools let agents:
- Read their own memory files (user taste, project context, skills)
- Write/update their memory based on observations
- Learn from interactions and improve over time
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..memory.agent_memory import get_agent_memory

logger = logging.getLogger(__name__)


class agent_memory_read:
    """Read from the agent's persistent memory."""

    @staticmethod
    def check_fn() -> bool:
        return True

    @staticmethod
    def get_schema() -> dict[str, Any]:
        return {
            "name": "agent_memory_read",
            "description": "Read from the agent's persistent memory files (user_taste, project_context, skill_journal, self_reflection, tool_usage_patterns).",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "enum": ["user_taste", "project_context", "skill_journal", "self_reflection", "tool_usage_patterns", "all"],
                        "description": "Which memory file to read. 'all' returns everything.",
                    },
                },
                "required": ["key"],
            },
        }

    @staticmethod
    def static_call(key: str, agent_id: str = "default") -> str:
        memory = get_agent_memory(agent_id)
        if key == "all":
            return json.dumps(memory.read_all(), ensure_ascii=False, indent=2)
        return memory.read(key)


class agent_memory_write:
    """Write to the agent's persistent memory."""

    @staticmethod
    def check_fn() -> bool:
        return True

    @staticmethod
    def get_schema() -> dict[str, Any]:
        return {
            "name": "agent_memory_write",
            "description": "Write or overwrite a memory file. Use this to update the agent's knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "enum": ["user_taste", "project_context", "skill_journal", "self_reflection", "tool_usage_patterns"],
                    },
                    "content": {
                        "type": "string",
                        "description": "Full content to write. Can use Markdown.",
                    },
                    "append": {
                        "type": "boolean",
                        "description": "If true, append instead of overwrite.",
                        "default": False,
                    },
                },
                "required": ["key", "content"],
            },
        }

    @staticmethod
    def static_call(key: str, content: str, append: bool = False, agent_id: str = "default") -> str:
        memory = get_agent_memory(agent_id)
        if append:
            memory.append(key, content)
        else:
            memory.write(key, content)
        return f"Memory '{key}' updated."


class agent_memory_learn:
    """Learn from an interaction and update memory."""

    @staticmethod
    def check_fn() -> bool:
        return True

    @staticmethod
    def get_schema() -> dict[str, Any]:
        return {
            "name": "agent_memory_learn",
            "description": "Record a learning from the current interaction. Update user taste, tool patterns, or propose a skill.",
            "parameters": {
                "type": "object",
                "properties": {
                    "observation": {
                        "type": "string",
                        "description": "What was observed. E.g., 'User prefers concise responses' or 'ReadFile + patch is better than write_file for edits'.",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["user_taste", "tool_pattern", "skill_proposal", "self_reflection"],
                        "description": "What kind of learning this is.",
                    },
                    "importance": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "default": "medium",
                    },
                },
                "required": ["observation", "category"],
            },
        }

    @staticmethod
    def static_call(
        observation: str,
        category: str,
        importance: str = "medium",
        agent_id: str = "default",
    ) -> str:
        memory = get_agent_memory(agent_id)

        if category == "user_taste":
            memory.update_user_taste(observation)
            return f"User taste updated: {observation}"

        elif category == "tool_pattern":
            memory.append("tool_usage_patterns", f"- {observation}")
            return f"Tool pattern recorded: {observation}"

        elif category == "skill_proposal":
            # Extract a name from the observation
            name = observation.split(".")[0][:40]
            memory.propose_skill(name, observation, "observed")
            return f"Skill proposed: {name}"

        elif category == "self_reflection":
            memory.append("self_reflection", f"- {observation}")
            return f"Self-reflection recorded: {observation}"

        return f"Unknown category: {category}"


class agent_memory_sync_context:
    """Sync project context (AGENTS.md, XERXES.md, etc.) into memory."""

    @staticmethod
    def check_fn() -> bool:
        return True

    @staticmethod
    def get_schema() -> dict[str, Any]:
        return {
            "name": "agent_memory_sync_context",
            "description": "Read AGENTS.md, XERXES.md, USER.md, SOUL.md from the current project and update the agent's project_context memory.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }

    @staticmethod
    def static_call(agent_id: str = "default") -> str:
        memory = get_agent_memory(agent_id)
        memory.sync_project_context()
        return "Project context synced to agent memory."
