# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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

"""Task runner for the daemon.

Wraps the streaming agent loop in a synchronous function suitable for
:class:`concurrent.futures.ThreadPoolExecutor`. Each task gets its own
:class:`~calute.streaming.events.AgentState` and streams events via an
optional callback for real-time progress to WebSocket clients.
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
    id: str
    prompt: str
    source: str = ""
    status: str = "pending"  # pending, running, completed, failed, cancelled
    result: str = ""
    error: str = ""
    created_at: str = ""
    completed_at: str = ""
    _cancel: bool = False

    def cancel(self) -> None:
        self._cancel = True


def create_task(prompt: str, source: str = "") -> Task:
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
    """Run a single agent task synchronously (meant for ThreadPoolExecutor).

    Args:
        task: The task to execute.
        config: LLM config (model, base_url, api_key, sampling params).
        system_prompt: System prompt for the agent.
        tool_executor: Tool executor callable.
        tool_schemas: Tool schemas for the LLM.
        on_event: Optional callback for streaming events to clients.
            Called as on_event(event_type, data_dict).

    Returns:
        The agent's final text response.
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
