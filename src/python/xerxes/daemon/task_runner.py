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
"""Task data model and synchronous task execution for the daemon.

``Task`` tracks metadata and cancellation state. ``create_task`` builds a new
instance, and ``run_task`` executes the agent loop in a blocking manner so it
can be offloaded to a thread pool.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..streaming.events import AgentState, TextChunk, ToolEnd, ToolStart
from ..streaming.loop import run as run_agent_loop


@dataclass
class Task:
    """Represents a single unit of work submitted to the daemon.

    Attributes:
        id (str): IN: Unique identifier (generated). OUT: Used for lookups.
        prompt (str): IN: User message text. OUT: Passed to the agent loop.
        source (str): IN: Origin label (e.g. ``"ws:..."`` or ``"socket"``).
            OUT: Stored for logging and listing.
        status (str): IN: Initial status. OUT: Updated by ``run_task``.
        result (str): IN: Empty initially. OUT: Filled on completion.
        error (str): IN: Empty initially. OUT: Filled on failure.
        created_at (str): IN: ISO timestamp at creation. OUT: Set by
            ``create_task``.
        completed_at (str): IN: Empty initially. OUT: Set on finish.
        _cancel (bool): IN: ``False`` initially. OUT: Set to ``True`` by
            ``cancel()`` to signal the loop to stop.
    """

    id: str
    prompt: str
    source: str = ""
    status: str = "pending"
    result: str = ""
    error: str = ""
    created_at: str = ""
    completed_at: str = ""
    _cancel: bool = False

    def cancel(self) -> None:
        """Signal that this task should be aborted.

        Returns:
            None: OUT: Sets ``_cancel`` to ``True``.
        """
        self._cancel = True


def create_task(prompt: str, source: str = "") -> Task:
    """Instantiate a new ``Task`` with a generated ID and timestamp.

    Args:
        prompt (str): IN: User message text. OUT: Stored on the task.
        source (str): IN: Origin label. OUT: Stored on the task.

    Returns:
        Task: OUT: Fresh task in ``"pending"`` status.
    """
    return Task(
        id=str(uuid.uuid4())[:8],
        prompt=prompt,
        source=source,
        created_at=datetime.now(UTC).isoformat(),
    )


def run_task(
    task: Task,
    config: dict[str, Any],
    system_prompt: str,
    tool_executor: Any = None,
    tool_schemas: list[dict[str, Any]] | None = None,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> str:
    """Execute the agent loop for a task and collect the result.

    This function is blocking and intended to run inside a thread-pool
    executor.

    Args:
        task (Task): IN: The task to execute. OUT: Mutated with status,
            result, error, and completion timestamp.
        config (dict[str, Any]): IN: Runtime configuration (model, API keys,
            etc.). OUT: Passed to ``run_agent_loop``.
        system_prompt (str): IN: System prompt text. OUT: Passed to the agent
            loop.
        tool_executor (Any): IN: Callable registry for tool execution. OUT:
            Passed to ``run_agent_loop``.
        tool_schemas (list[dict[str, Any]] | None): IN: JSON schemas for
            available tools. OUT: Passed to ``run_agent_loop``.
        on_event (Callable[[str, dict[str, Any]], None] | None): IN: Optional
            callback for streaming events (``task.started``,
            ``task.progress``, etc.). OUT: Invoked as events are produced.

    Returns:
        str: OUT: Full agent response text, or an error message on failure.
    """

    task.status = "running"
    if on_event:
        on_event("task.started", {"task_id": task.id, "prompt": task.prompt})

    state = AgentState()
    output_parts: list[str] = []

    try:
        for event in run_agent_loop(
            user_message=task.prompt,
            state=state,
            config=config,
            system_prompt=system_prompt,
            tool_executor=tool_executor,
            tool_schemas=tool_schemas,
            cancel_check=lambda: task._cancel,
        ):
            if isinstance(event, TextChunk):
                output_parts.append(event.text)
                if on_event:
                    on_event("task.progress", {"task_id": task.id, "text": event.text})

            elif isinstance(event, ToolStart):
                if on_event:
                    on_event(
                        "task.tool",
                        {
                            "task_id": task.id,
                            "name": event.name,
                            "inputs": event.inputs,
                        },
                    )

            elif isinstance(event, ToolEnd):
                if on_event:
                    on_event(
                        "task.tool_done",
                        {
                            "task_id": task.id,
                            "name": event.name,
                            "permitted": event.permitted,
                            "duration_ms": event.duration_ms,
                        },
                    )

        result = "".join(output_parts)
        task.status = "completed"
        task.result = result
        task.completed_at = datetime.now(UTC).isoformat()

        if on_event:
            on_event("task.completed", {"task_id": task.id, "result": result})

        return result

    except Exception as exc:
        task.status = "failed"
        task.error = str(exc)
        task.completed_at = datetime.now(UTC).isoformat()

        if on_event:
            on_event("task.failed", {"task_id": task.id, "error": str(exc)})

        return f"Error: {exc}"
