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
"""Background subagent handles and their lifecycle controller.

Each handle wraps a cloned :class:`Agent` plus a single :class:`asyncio.Task`
that drives ``xerxes.create_response``. The manager exposes spawn / send /
wait / close / resume primitives so the parent agent can fan work out to
short-lived helpers while staying in control of their lifecycle.
"""

from __future__ import annotations

import asyncio
import typing as tp
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..runtime.profiles import PromptProfile
from ..types import Agent, ResponseResult


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""

    return datetime.now(UTC).isoformat()


@dataclass
class SpawnedAgentHandle:
    """Reference to one in-flight (or recently finished) subagent.

    A handle is created by :meth:`SpawnedAgentManager.spawn` and lives until
    the parent agent explicitly closes it. The ``task`` field holds the
    coroutine currently running on the subagent (or ``None`` between jobs);
    pending follow-up messages buffer in ``queue`` so the parent can pipeline
    work without waiting between calls.

    Attributes:
        handle_id: Public alias used by ``send_input`` / ``wait`` / ``close``.
        agent: Cloned :class:`Agent` driven by this handle.
        source_agent_id: Identifier of the parent agent that spawned us.
        status: Lifecycle marker — ``idle``, ``running``, ``completed``,
            ``cancelled``, ``interrupted``, ``error``, or ``closed``.
        created_at: ISO-8601 spawn timestamp.
        updated_at: ISO-8601 timestamp of the last state change.
        prompt_profile: Prompt profile name applied to subagent turns.
        last_input: Most recent message dispatched to the subagent.
        last_output: Most recent textual response produced by the subagent.
        error: Stringified exception when ``status == "error"``.
        queue: FIFO of follow-up messages waiting for the active task.
        task: Currently running asyncio task, or ``None``.
        closed: ``True`` after :meth:`SpawnedAgentManager.close`; the handle
            still exists for inspection but rejects new work until resumed.
    """

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
        """Return a wire-safe view of the handle for tool responses."""

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
    """Lifecycle controller for the operator's spawned subagent handles.

    The manager keeps a flat registry of :class:`SpawnedAgentHandle`
    entries. Each subagent reuses the parent's Xerxes orchestrator but
    runs through ``create_response`` with its own prompt profile override
    so it cannot accidentally inherit the parent's tool budget.
    """

    def __init__(self, xerxes: tp.Any, runtime_state: tp.Any) -> None:
        """Bind the manager to a running Xerxes instance and runtime state.

        Args:
            xerxes: The Xerxes orchestrator used to clone agents and drive
                ``create_response``.
            runtime_state: Streaming runtime state — used to read and write
                per-agent overrides (prompt profile, etc.).
        """

        self._xerxes = xerxes
        self._runtime_state = runtime_state
        self._handles: dict[str, SpawnedAgentHandle] = {}

    def list_handles(self) -> list[dict[str, tp.Any]]:
        """Return a snapshot list of every tracked subagent handle."""

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
        """Create a new background subagent handle and optionally seed work.

        The new handle clones the parent agent (or the agent named by
        ``agent_id``), registers a prompt-profile override on the runtime
        state, and — if an initial ``message`` (or legacy
        ``task_description``) is supplied — dispatches that message
        immediately via :meth:`send_input`.

        Args:
            message: First message to send to the subagent.
            task_description: Legacy alias for ``message``.
            agent_id: Parent agent to clone; defaults to the currently
                active agent.
            prompt_profile: Prompt profile applied to the subagent's turns;
                defaults to :attr:`PromptProfile.MINIMAL`.
            nickname: Optional stable handle id (defaults to a random one).
        """

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
        """Hand more work to an existing subagent.

        Resolves the target handle (most-recent open handle if ``handle_id``
        is ``None``), validates that it is still open, and either:

        * Starts a fresh task if nothing is currently running.
        * Queues the message behind the active task when ``interrupt=False``.
        * Cancels the active task and starts a new one when ``interrupt=True``.

        Args:
            handle_id: Target handle; defaults to the most recently
                updated open handle.
            message: Body of work to dispatch.
            task_description: Legacy alias for ``message``.
            interrupt: Cancel the current task before starting the new one.
        """

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
        """Wait for the given subagents to settle or the timeout to expire.

        Splits the targets into ``completed`` (task finished or never had
        one) and ``pending`` (still running when the timeout hit).
        """

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
        """Reopen a previously closed handle so it can take new work.

        Clears the ``closed`` flag and resets the status from ``"closed"``
        back to ``"idle"``. Other statuses are left intact.
        """

        handle = self._require_handle(handle_id)
        handle.closed = False
        if handle.status == "closed":
            handle.status = "idle"
        handle.updated_at = _now_iso()
        return handle.snapshot()

    def close(self, handle_id: str) -> dict[str, tp.Any]:
        """Cancel the active task (if any) and mark the handle closed.

        Closed handles remain registered so subsequent ``resume`` or
        ``wait`` calls still find them. The returned snapshot includes
        ``previous_status`` so callers can see what we cancelled.
        """

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
        """Drive one ``create_response`` call for the given handle.

        Updates the handle's lifecycle fields based on the outcome
        (``completed`` / ``cancelled`` / ``error``) and, if more messages
        are queued and the handle is still open, chains the next task.
        """

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
        """Return the named handle or raise ``ValueError`` if unknown."""

        if handle_id not in self._handles:
            raise ValueError(f"Spawned agent not found: {handle_id}")
        return self._handles[handle_id]

    def _resolve_handle_id(self, handle_id: str | None) -> str:
        """Pick the most-recently-updated open handle when no id is given."""

        if handle_id:
            return handle_id

        open_handles = [handle for handle in self._handles.values() if not handle.closed]
        if not open_handles:
            raise ValueError("Spawned agent target is required because no open handles exist")

        latest = max(open_handles, key=lambda handle: (handle.updated_at, handle.created_at))
        return latest.handle_id
