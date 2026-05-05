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
"""Subagents module for Xerxes.

Exports:
    - SpawnedAgentHandle
    - SpawnedAgentManager"""

from __future__ import annotations

import asyncio
import typing as tp
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..runtime.profiles import PromptProfile
from ..types import Agent, ResponseResult


def _now_iso() -> str:
    """Internal helper to now iso.

    Returns:
        str: OUT: Result of the operation."""

    return datetime.now(UTC).isoformat()


@dataclass
class SpawnedAgentHandle:
    """Spawned agent handle.

    Attributes:
        handle_id (str): handle id.
        agent (Agent): agent.
        source_agent_id (str | None): source agent id.
        status (str): status.
        created_at (str): created at.
        updated_at (str): updated at.
        prompt_profile (str): prompt profile.
        last_input (str | None): last input.
        last_output (str | None): last output.
        error (str | None): error.
        queue (list[str]): queue.
        task (asyncio.Task | None): task.
        closed (bool): closed."""

    handle_id: str
    agent: Agent
    source_agent_id: str | None
    status: str = "idle"
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    prompt_profile: str = PromptProfile.MINIMAL.value
    last_input: str | None = None
    last_output: str | None = None
    error: str | None = None
    queue: list[str] = field(default_factory=list)
    task: asyncio.Task | None = None
    closed: bool = False

    def snapshot(self) -> dict[str, tp.Any]:
        """Snapshot.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        return {
            "id": self.handle_id,
            "name": self.agent.name or self.handle_id,
            "agent_id": self.agent.id,
            "source_agent_id": self.source_agent_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "prompt_profile": self.prompt_profile,
            "last_input": self.last_input,
            "last_output": self.last_output,
            "error": self.error,
            "queue_size": len(self.queue),
            "queued_preview": self.queue[0] if self.queue else None,
            "closed": self.closed,
        }


class SpawnedAgentManager:
    """Spawned agent manager."""

    def __init__(self, xerxes: tp.Any, runtime_state: tp.Any) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            xerxes (tp.Any): IN: xerxes. OUT: Consumed during execution.
            runtime_state (tp.Any): IN: runtime state. OUT: Consumed during execution."""

        self._xerxes = xerxes
        self._runtime_state = runtime_state
        self._handles: dict[str, SpawnedAgentHandle] = {}

    def list_handles(self) -> list[dict[str, tp.Any]]:
        """List handles.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[dict[str, tp.Any]]: OUT: Result of the operation."""

        return [handle.snapshot() for handle in self._handles.values()]

    async def spawn(
        self,
        *,
        message: str | None = None,
        task_description: str | None = None,
        agent_id: str | None = None,
        prompt_profile: str | None = None,
        nickname: str | None = None,
    ) -> dict[str, tp.Any]:
        """Asynchronously Spawn.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            message (str | None, optional): IN: message. Defaults to None. OUT: Consumed during execution.
            task_description (str | None, optional): IN: task description. Defaults to None. OUT: Consumed during execution.
            agent_id (str | None, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
            prompt_profile (str | None, optional): IN: prompt profile. Defaults to None. OUT: Consumed during execution.
            nickname (str | None, optional): IN: nickname. Defaults to None. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        source_agent = (
            self._xerxes.orchestrator.agents[agent_id] if agent_id else self._xerxes.orchestrator.get_current_agent()
        )
        handle_id = nickname or f"subagent_{uuid.uuid4().hex[:10]}"
        cloned = source_agent.model_copy(deep=False)
        cloned.id = handle_id
        cloned.name = nickname or cloned.name or handle_id
        resolved_profile = prompt_profile or PromptProfile.MINIMAL.value
        handle = SpawnedAgentHandle(
            handle_id=handle_id,
            agent=cloned,
            source_agent_id=source_agent.id,
            prompt_profile=resolved_profile,
        )
        self._handles[handle_id] = handle

        overrides = self._runtime_state.config.agent_overrides.setdefault(
            handle_id,
            self._runtime_state.get_agent_overrides(handle_id),
        )
        overrides.prompt_profile = resolved_profile

        initial_message = message if message is not None else task_description
        if initial_message:
            await self.send_input(handle_id, message=initial_message, interrupt=False)
        return handle.snapshot()

    async def send_input(
        self,
        handle_id: str | None = None,
        *,
        message: str | None = None,
        task_description: str | None = None,
        interrupt: bool = False,
    ) -> dict[str, tp.Any]:
        """Asynchronously Send input.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            handle_id (str | None, optional): IN: handle id. Defaults to None. OUT: Consumed during execution.
            message (str | None, optional): IN: message. Defaults to None. OUT: Consumed during execution.
            task_description (str | None, optional): IN: task description. Defaults to None. OUT: Consumed during execution.
            interrupt (bool, optional): IN: interrupt. Defaults to False. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        resolved_handle_id = self._resolve_handle_id(handle_id)
        resolved_message = message if message is not None else task_description
        if resolved_message is None:
            raise ValueError("Spawned agent input is required")

        handle = self._require_handle(resolved_handle_id)
        if handle.closed:
            raise ValueError(f"Spawned agent is closed: {resolved_handle_id}")

        if handle.task is not None and not handle.task.done():
            if interrupt:
                handle.task.cancel()
                handle.status = "interrupted"
            else:
                handle.queue.append(resolved_message)
                handle.updated_at = _now_iso()
                return handle.snapshot()

        handle.task = asyncio.create_task(self._run_handle(handle, resolved_message))
        return handle.snapshot()

    async def wait(self, targets: list[str], timeout_ms: int = 30000) -> dict[str, tp.Any]:
        """Asynchronously Wait.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            targets (list[str]): IN: targets. OUT: Consumed during execution.
            timeout_ms (int, optional): IN: timeout ms. Defaults to 30000. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        handles = [self._require_handle(target) for target in targets]
        tasks = [handle.task for handle in handles if handle.task is not None]
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=max(timeout_ms, 0) / 1000)
        else:
            done, pending = set(), set()
        return {
            "completed": [handle.snapshot() for handle in handles if handle.task in done or handle.task is None],
            "pending": [handle.snapshot() for handle in handles if handle.task in pending],
        }

    def resume(self, handle_id: str) -> dict[str, tp.Any]:
        """Resume.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            handle_id (str): IN: handle id. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        handle = self._require_handle(handle_id)
        handle.closed = False
        if handle.status == "closed":
            handle.status = "idle"
        handle.updated_at = _now_iso()
        return handle.snapshot()

    def close(self, handle_id: str) -> dict[str, tp.Any]:
        """Close.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            handle_id (str): IN: handle id. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        handle = self._require_handle(handle_id)
        previous_status = handle.status
        if handle.task is not None and not handle.task.done():
            handle.task.cancel()
        handle.closed = True
        handle.status = "closed"
        handle.updated_at = _now_iso()
        out = handle.snapshot()
        out["previous_status"] = previous_status
        return out

    async def _run_handle(self, handle: SpawnedAgentHandle, message: str) -> None:
        """Asynchronously Internal helper to run handle.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            handle (SpawnedAgentHandle): IN: handle. OUT: Consumed during execution.
            message (str): IN: message. OUT: Consumed during execution."""

        handle.status = "running"
        handle.last_input = message
        handle.updated_at = _now_iso()
        try:
            response = await self._xerxes.create_response(
                prompt=message,
                agent_id=handle.agent,
                stream=False,
                apply_functions=True,
            )
            if isinstance(response, ResponseResult):
                handle.last_output = response.content
            else:
                handle.last_output = getattr(response, "content", str(response))
            handle.status = "completed"
            handle.error = None
        except asyncio.CancelledError:
            handle.status = "cancelled"
            handle.error = "cancelled"
            raise
        except Exception as exc:
            handle.status = "error"
            handle.error = str(exc)
        finally:
            handle.updated_at = _now_iso()
            if handle.queue and not handle.closed:
                next_message = handle.queue.pop(0)
                handle.task = asyncio.create_task(self._run_handle(handle, next_message))

    def _require_handle(self, handle_id: str) -> SpawnedAgentHandle:
        """Internal helper to require handle.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            handle_id (str): IN: handle id. OUT: Consumed during execution.
        Returns:
            SpawnedAgentHandle: OUT: Result of the operation."""

        if handle_id not in self._handles:
            raise ValueError(f"Spawned agent not found: {handle_id}")
        return self._handles[handle_id]

    def _resolve_handle_id(self, handle_id: str | None) -> str:
        """Internal helper to resolve handle id.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            handle_id (str | None): IN: handle id. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        if handle_id:
            return handle_id

        open_handles = [handle for handle in self._handles.values() if not handle.closed]
        if not open_handles:
            raise ValueError("Spawned agent target is required because no open handles exist")

        latest = max(open_handles, key=lambda handle: (handle.updated_at, handle.created_at))
        return latest.handle_id
