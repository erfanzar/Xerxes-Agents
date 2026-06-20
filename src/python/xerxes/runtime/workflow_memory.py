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
"""Runtime capture for explicit user workflow memory.

This is a guardrail around the model, not a replacement for the agent-facing
``agent_memory_*`` tools. When the user explicitly tells the agent to remember
workflow facts for the current project, the runtime persists that note before
the model has a chance to forget or skip the memory tool call.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..tools.agent_memory_tool import active_memory

WORKFLOW_MEMORY_FILE = "WORKFLOW.md"
EXPLICIT_MEMORY_MARKERS = (
    "remember",
    "keep in memory",
    "save this",
    "note this",
    "note that",
    "for your memory",
    "my workflow",
    "real workflow",
)
PROJECT_WORKFLOW_MARKERS = (
    "workflow",
    "big project",
    "big projects",
    "large project",
    "large projects",
    "large repo",
    "large repos",
    "codebase",
    "codebases",
    "project",
)
PROJECT_MEMORY_INTENTS = (
    "remember",
    "save",
    "learn",
    "understand",
    "use this",
    "always know",
    "keep track",
)


@dataclass(frozen=True)
class WorkflowMemoryCapture:
    """Result of a runtime workflow-memory capture attempt."""

    captured: bool
    reason: str = ""
    scope: str = ""
    path: str = ""


def capture_user_workflow_memory(user_message: str, *, project_root: Path | None = None) -> WorkflowMemoryCapture:
    """Persist explicit workflow-memory user messages into project memory.

    Args:
        user_message: Raw top-level user prompt for this turn.
        project_root: Current project root, used only for human-readable
            context in the saved note.

    Returns:
        Capture result. Non-memory-looking prompts are a no-op.
    """
    message = user_message.strip()
    if not message:
        return WorkflowMemoryCapture(False, "empty")
    if not should_capture_workflow_memory(message):
        return WorkflowMemoryCapture(False, "no_signal")

    memory = active_memory()
    if memory is None:
        return WorkflowMemoryCapture(False, "memory_unavailable")

    scope = "project" if memory.has_project_scope() else "global"
    body = _format_workflow_note(message, project_root=project_root)
    try:
        existing = memory.read(scope, WORKFLOW_MEMORY_FILE)
    except FileNotFoundError:
        existing = "# Workflow Memory\n\nDurable user workflow instructions and project operating notes.\n"
    except ValueError as exc:
        return WorkflowMemoryCapture(False, str(exc))

    if message in existing:
        return WorkflowMemoryCapture(False, "duplicate")

    if existing.strip():
        existing = existing.rstrip() + "\n\n"
    result = memory.write(scope, WORKFLOW_MEMORY_FILE, existing + body + "\n")
    return WorkflowMemoryCapture(True, scope=scope, path=result["path"])


def should_capture_workflow_memory(message: str) -> bool:
    """Return true when ``message`` carries an explicit durable-memory signal."""
    lowered = message.lower()
    if any(marker in lowered for marker in EXPLICIT_MEMORY_MARKERS):
        return True
    if "want" not in lowered:
        return False
    return any(marker in lowered for marker in PROJECT_WORKFLOW_MARKERS) and any(
        intent in lowered for intent in PROJECT_MEMORY_INTENTS
    )


def _format_workflow_note(message: str, *, project_root: Path | None) -> str:
    root = str(project_root.expanduser()) if project_root is not None else ""
    timestamp = datetime.now(UTC).isoformat()
    project_line = f"\n**Project root:** `{root}`" if root else ""
    return f"## {timestamp} - user workflow note{project_line}\n\n**Instruction:** {message.strip()}"
