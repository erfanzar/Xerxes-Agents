# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.

"""Background sub-agent manager for operator tooling.

Provides :class:`SpawnedAgentManager`, which creates, tracks, and
orchestrates background Xerxes sub-agent handles.  Each handle wraps a
cloned :class:`~xerxes.types.Agent` and can be sent work, waited on,
interrupted, resumed, or closed independently.
"""

from __future__ import annotations

import asyncio
import typing as tp
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, UTC

from ..runtime.profiles import PromptProfile
from ..types import Agent, ResponseResult


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string.

    Returns:
        A timezone-aware ISO 8601 timestamp string.
    """
    return datetime.now(UTC).isoformat()


@dataclass
class SpawnedAgentHandle:
    """State for a spawned background agent.

    Each handle tracks the lifecycle of one sub-agent, including its
    current status, message queue, last input/output, and the
    :class:`asyncio.Task` running its work.

    Attributes:
        handle_id: Unique identifier for this handle, used in all
            operator tool calls that reference the sub-agent.
        agent: The cloned :class:`~xerxes.types.Agent` instance that
            processes work for this handle.
        source_agent_id: Identifier of the parent agent that this
            handle was spawned from.  ``None`` when unknown.
        status: Current lifecycle status.  One of ``"idle"``,
            ``"running"``, ``"completed"``, ``"error"``,
            ``"interrupted"``, ``"cancelled"``, or ``"closed"``.
        created_at: ISO 8601 timestamp of when the handle was created.
        updated_at: ISO 8601 timestamp of the most recent state change.
        prompt_profile: Name of the prompt profile applied to the
            sub-agent.
        last_input: The most recent message sent to this handle.
        last_output: The most recent response produced by the
            sub-agent.
        error: Error message string if the last run failed.
        queue: FIFO queue of messages waiting to be processed after
            the current task finishes.
        task: The :class:`asyncio.Task` running the current work, or
            ``None`` when idle.
        closed: Whether the handle has been explicitly closed.
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
        """Return a serialisable handle summary.

        Produces a dictionary suitable for JSON serialisation and tool
        result payloads, omitting the heavy ``agent`` and ``task``
        objects.

        Returns:
            A dictionary with all scalar state fields plus derived
            values like ``queue_size`` and ``queued_preview``.
        """
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
    """Manage background Xerxes sub-agent handles.

    Maintains a registry of :class:`SpawnedAgentHandle` instances and
    provides methods to spawn new handles, send them work, wait for
    completion, resume, and close them.

    Attributes:
        _xerxes: Reference to the parent :class:`Xerxes` instance used
            for creating responses on behalf of sub-agents.
        _runtime_state: Reference to the shared runtime state, used
            for agent override configuration.
        _handles: Internal mapping from ``handle_id`` to the
            corresponding :class:`SpawnedAgentHandle`.
    """

    def __init__(self, xerxes: tp.Any, runtime_state: tp.Any) -> None:
        """Initialise the manager with references to the parent runtime.

        Args:
            xerxes: The parent :class:`Xerxes` instance that will
                execute sub-agent responses.
            runtime_state: The shared runtime state holding
                configuration and policy data.
        """
        self._xerxes = xerxes
        self._runtime_state = runtime_state
        self._handles: dict[str, SpawnedAgentHandle] = {}

    def list_handles(self) -> list[dict[str, tp.Any]]:
        """Return summaries for all spawned agent handles.

        Returns:
            A list of snapshot dictionaries, one per handle.
        """
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
        """Create a background handle and optionally start the first task.

        Clones the source agent, registers the handle, configures the
        prompt profile override, and optionally queues the initial
        message for immediate processing.

        Args:
            message: Optional initial instruction to send to the
                sub-agent immediately after spawning.
            task_description: Backward-compatible alias for
                ``message`` used by older callers.
            agent_id: Registered agent ID to clone.  When ``None``,
                the orchestrator's current/default agent is used.
            prompt_profile: Prompt profile name override.  Defaults to
                :attr:`PromptProfile.MINIMAL`.
            nickname: Human-readable label for the handle.  Also used
                as the ``handle_id`` when provided.

        Returns:
            The snapshot dictionary of the newly created handle.

        Raises:
            KeyError: If ``agent_id`` does not match a registered
                agent.
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
        """Queue or interrupt a spawned agent task.

        If the handle has a running task and ``interrupt`` is ``False``,
        the message is appended to the handle's FIFO queue.  If
        ``interrupt`` is ``True``, the running task is cancelled before
        the new message is dispatched.

        Args:
            handle_id: Identifier of the target handle.  When omitted,
                the most recently updated non-closed handle is used.
            message: Instruction text to deliver to the sub-agent.
            task_description: Backward-compatible alias for
                ``message`` used by older callers.
            interrupt: When ``True``, cancel any running task and
                process this message immediately.

        Returns:
            The updated handle snapshot dictionary.

        Raises:
            ValueError: If the handle is not found or has been closed.
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
        """Wait for the given spawned agent handles to finish or timeout.

        Collects the :class:`asyncio.Task` objects for the specified
        handles and waits for them using :func:`asyncio.wait`.

        Args:
            targets: List of handle IDs to wait on.
            timeout_ms: Maximum wait time in milliseconds.  Handles
                whose tasks are still running after this period appear
                in the ``pending`` list.

        Returns:
            A dictionary with two keys:

            - ``completed``: Snapshots of handles whose tasks finished
              (or that had no running task).
            - ``pending``: Snapshots of handles still running at
              timeout.

        Raises:
            ValueError: If any target handle ID is not found.
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
        """Re-open a closed spawned-agent handle.

        Clears the ``closed`` flag and resets the status to ``"idle"``
        if it was ``"closed"``.

        Args:
            handle_id: Identifier of the handle to resume.

        Returns:
            The updated handle snapshot dictionary.

        Raises:
            ValueError: If the handle ID is not found.
        """
        handle = self._require_handle(handle_id)
        handle.closed = False
        if handle.status == "closed":
            handle.status = "idle"
        handle.updated_at = _now_iso()
        return handle.snapshot()

    def close(self, handle_id: str) -> dict[str, tp.Any]:
        """Close a spawned-agent handle and cancel any running task.

        Marks the handle as closed, cancels its active task if one
        exists, and records the previous status in the returned
        snapshot.

        Args:
            handle_id: Identifier of the handle to close.

        Returns:
            The final handle snapshot dictionary with an additional
            ``previous_status`` key indicating the status before
            closure.

        Raises:
            ValueError: If the handle ID is not found.
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
        """Execute a single task on behalf of a spawned agent handle.

        Sets the handle status to ``"running"``, invokes
        :meth:`Xerxes.create_response`, and updates the handle with
        the result or error.  On completion, if queued messages remain
        and the handle is not closed, the next message is dispatched
        automatically.

        Args:
            handle: The :class:`SpawnedAgentHandle` to run.
            message: The instruction text to process.
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
        """Return a tracked handle or raise.

        Args:
            handle_id: Identifier of the handle to look up.

        Returns:
            The :class:`SpawnedAgentHandle` instance.

        Raises:
            ValueError: If no handle with the given ID exists.
        """
        if handle_id not in self._handles:
            raise ValueError(f"Spawned agent not found: {handle_id}")
        return self._handles[handle_id]

    def _resolve_handle_id(self, handle_id: str | None) -> str:
        """Resolve an explicit or implicit spawned-agent handle identifier.

        When ``handle_id`` is omitted, the most recently updated
        non-closed handle is selected. This keeps operator flows
        working when the model omits the handle immediately after a
        ``spawn_agent`` call.
        """
        if handle_id:
            return handle_id

        open_handles = [handle for handle in self._handles.values() if not handle.closed]
        if not open_handles:
            raise ValueError("Spawned agent target is required because no open handles exist")

        latest = max(open_handles, key=lambda handle: (handle.updated_at, handle.created_at))
        return latest.handle_id
