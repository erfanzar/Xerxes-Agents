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
"""Subagent lifecycle: spawn, message, await, monitor, cancel."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from ...types import AgentBaseFn

_agent_manager = None


def _default_subagent_base() -> str:
    """Build a scoped delegate-context base prompt for a spawned subagent.

    A bare "You are a helpful AI assistant." leaves delegated work context-blind:
    no project facts, no memory, no cwd. This layers in delegate etiquette, the
    parent's project/workspace context (when an active session is bound), the
    shared-memory note, and the working directory — without leaking the parent's
    full conversation. Best-effort: every lookup degrades gracefully.
    """
    parts: list[str] = [
        "You are a focused subagent spawned by a parent Xerxes agent to complete a "
        "single delegated subtask. You do NOT see the parent's conversation, so work "
        "only from the prompt you were given. Use your tools to take real action, "
        "verify your work, and return a COMPLETE, self-contained result the parent "
        "can use without a follow-up round-trip. You share this project's persistent "
        "memory: read what's relevant and record durable findings with the "
        "agent_memory_* tools.",
    ]
    try:
        from ...runtime.session_context import get_active_session

        session = get_active_session()
        if session is not None:
            ctx = session.workspace.load_context()
            if getattr(ctx, "prompt", ""):
                parts.append(ctx.prompt)
    except Exception:
        pass
    try:
        parts.append(f"Working directory: {os.getcwd()}")
    except Exception:
        pass
    return "\n\n".join(parts)


def _build_subagent_system_prompt(
    base: str | None = None,
    *,
    include_active_skills: bool = False,
) -> str:
    """Build system prompt for spawned subagents.

    Args:
        base: Base system prompt. When ``None``, a scoped delegate-context base
            (project context + memory note + cwd) is built via
            :func:`_default_subagent_base` so subagents aren't context-blind.
        include_active_skills: Whether to include active skills in prompt.

    Returns:
        Complete system prompt for subagent.
    """
    if base is None:
        base = _default_subagent_base()
    if not include_active_skills:
        return base

    from ...extensions.skills import get_active_skills
    from ...tools.agent_meta_tools import _skill_registry

    active = get_active_skills()
    if not active or _skill_registry is None:
        return base

    sections: list[str] = []
    for name in active:
        skill = _skill_registry.get(name)
        if skill is not None:
            sections.append(skill.to_prompt_section())

    if not sections:
        return base

    return "\n\n".join(sections) + "\n\n" + base


def _get_agent_manager():
    """Get or create the global agent manager singleton.

    Returns:
        The SubAgentManager instance.
    """
    global _agent_manager
    if _agent_manager is None:
        from ...agents.subagent_manager import SubAgentManager
        from ...runtime.bridge import build_tool_executor, populate_registry

        try:
            max_concurrent = int(os.environ.get("XERXES_SUBAGENT_MAX_CONCURRENT", "16"))
        except ValueError:
            max_concurrent = 16
        _agent_manager = SubAgentManager(max_concurrent=max(1, max_concurrent))
        registry = populate_registry()
        _agent_manager._tool_executor = build_tool_executor(registry=registry)
        _agent_manager._tool_schemas = registry.tool_schemas()
    return _agent_manager


def _parse_agents_payload(raw: str) -> list[dict[str, Any]] | str:
    """Best-effort parse of ``SpawnAgents.agents`` when the LLM hands us a string.

    LLMs routinely mis-escape the inner prompt of a structured ``agents``
    argument — single vs double quotes, smart quotes, leading/trailing
    code-fence markers, or a JSON-in-JSON wrapper. Rather than rejecting the
    first failure, try a few common cleanups before giving up; this avoids
    pushing the model back to per-prompt ``AgentTool`` calls (which then
    block sequentially) just because of a quoting hiccup. Returns the
    parsed list on success, or the original raw string when every strategy
    fails (so the caller can surface a precise error).
    """
    stripped = raw.strip()
    if not stripped:
        return raw

    # Strip code-fence wrapping (```json … ```) the LLM sometimes adds.
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            inner = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:])
            stripped = inner.strip()

    candidates = [stripped]
    # Normalise smart quotes to ASCII so json.loads stops choking on copy-pastes.
    smart_q = {"“": '"', "”": '"', "‘": "'", "’": "'"}  # noqa: RUF001 - keys are intentionally smart quotes
    norm = "".join(smart_q.get(ch, ch) for ch in stripped)
    if norm != stripped:
        candidates.append(norm)
    # Single-quoted JSON-ish — last-resort retry with quotes flipped.
    if "'" in norm and '"' not in norm:
        candidates.append(norm.replace("'", '"'))

    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    return raw


def _subagent_wait_timeout(value: float | int | str | None = None) -> float | None:
    """Return the bounded wait timeout for sub-agent joins.

    Args:
        value: Timeout value from caller or environment.

    Returns:
        Timeout in seconds, or None for no timeout.
    """
    raw = value
    if raw is None:
        raw = os.environ.get("XERXES_SUBAGENT_WAIT_TIMEOUT")
    if raw is None:
        raw = os.environ.get("XERXES_SPAWN_AGENTS_WAIT_TIMEOUT", "120")
    if raw is None:
        return 120.0
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return 120.0
    if timeout <= 0:
        return None
    return timeout


def _spawn_agents_wait_timeout(value: float | int | str | None = None) -> float | None:
    """Return the bounded wait timeout for SpawnAgents joins."""
    return _subagent_wait_timeout(value)


class AgentTool(AgentBaseFn):
    """Spawn a subagent to execute a task. Blocks by default — prefer the async pattern.

    For one quick delegation where you genuinely need the result before
    proceeding, ``AgentTool`` (with ``wait=True``) is fine. For anything
    longer, anything parallel, or anything you want to monitor mid-flight,
    the async pattern is strictly better:

        1. ``SpawnAgents(agents=[...], wait=False)`` — fire several at once,
           returns their ids immediately.
        2. Keep working, or ``AwaitAgents(agent_ids=[...], wake_on="any",
           timeout_seconds=N)`` to sleep until one finishes / user wakes us /
           timeout.
        3. ``PeekAgent("name")`` or ``CheckAgentMessages()`` to see progress.
        4. ``ResetAgent("name", new_prompt="…")`` if it drifted off-track,
           or ``TaskStopTool("name")`` to kill it outright.

    Example (one-off, blocking):
        >>> AgentTool.static_call(prompt="Run quick lint", wait=True)

    Example (preferred async):
        >>> SpawnAgents.static_call(agents=[{"name": "lint", "prompt": "..."}], wait=False)
        >>> AwaitAgents.static_call(agent_ids=["lint"], timeout_seconds=60)
    """

    @staticmethod
    def static_call(
        prompt: str,
        subagent_type: str = "general-purpose",
        isolation: str = "",
        name: str = "",
        model: str = "",
        wait: bool = True,
        timeout: float | None = None,
        **context_variables,
    ) -> str:
        """Spawn a subagent. Returns the result when ``wait=True``, or a running snapshot otherwise.

        Prefer ``SpawnAgents(wait=False) + AwaitAgents`` for parallel work or
        anything that might run more than ~15s — that pattern lets you
        monitor and intervene; ``wait=True`` blocks the whole main turn until
        the subagent finishes.

        Args:
            prompt: Task description for the subagent.
            subagent_type: Type of agent to spawn (e.g., 'coder', 'reviewer').
            isolation: Optional isolation mode (e.g., 'worktree').
            name: Optional name for the subagent (lets you address it later
                with PeekAgent / ResetAgent / TaskStopTool).
            model: Optional model override.
            wait: Whether to block until the subagent finishes. Defaults to
                True for backward compat — set ``False`` for true async spawn
                and use ``AwaitAgents`` to coordinate.
            timeout: Maximum seconds to wait before returning a running snapshot.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            The subagent's result or a JSON running-snapshot.
        """
        from ...agents.definitions import get_agent_definition
        from ...runtime.config_context import get_inheritable

        mgr = _get_agent_manager()
        agent_def = get_agent_definition(subagent_type)

        config: dict[str, Any] = get_inheritable()
        if model:
            config["model"] = model
        elif agent_def and agent_def.model:
            config["model"] = agent_def.model

        task = mgr.spawn(
            prompt=prompt,
            config=config,
            system_prompt=_build_subagent_system_prompt(),
            agent_def=agent_def,
            isolation=isolation,
            name=name,
        )

        # Skip the wait on any terminal status — "failed", "completed", and
        # "cancelled" are all already settled, so blocking on mgr.wait would
        # either return immediately (best case) or hang on a stale handle.
        # Previously the tuple held only ("failed",), which meant we still
        # blocked on already-completed tasks.
        if wait and task.status not in ("failed", "completed", "cancelled"):
            mgr.wait(task.id, timeout=_subagent_wait_timeout(timeout))

        if task.status == "completed" and task.result is not None:
            return task.result
        if task.status == "failed":
            return f"Agent failed: {task.error}"
        if task.status == "cancelled":
            return "[Sub-agent was cancelled.]"

        snapshot = task.snapshot()
        snapshot["note"] = "Agent is still running; use TaskGetTool or TaskListTool to check it later."
        return json.dumps(snapshot, indent=2, default=str)


class SendMessageTool(AgentBaseFn):
    """Send a message to a running subagent.

    Allows communication with running subagents for follow-up tasks.

    Example:
        >>> SendMessageTool.static_call(target="my-agent", message="Continue with next step")
    """

    @staticmethod
    def static_call(
        target: str,
        message: str,
        **context_variables,
    ) -> str:
        """Send a message to a running subagent.

        Args:
            target: Agent name or task ID.
            message: Message content to send.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Confirmation or error message.
        """
        mgr = _get_agent_manager()
        task_id = mgr._by_name.get(target, target)
        task = mgr.tasks.get(task_id)
        if task is None:
            return f"Error: agent '{target}' not found."
        if task.status not in ("running", "pending"):
            return (
                f"Error: agent '{target}' is already {task.status}. "
                f"Spawn agents with wait=False if you need to send follow-up messages. "
                f"Current result: {str(task.result)[:500]}"
            )
        ok = mgr.send_message(target, message)
        if ok:
            return f"Message queued for agent '{target}'."
        return f"Error: agent '{target}' not found or already completed."


class TaskCreateTool(AgentBaseFn):
    """Create a subagent task without waiting for completion.

    Spawns an agent that runs in the background for later retrieval.

    Example:
        >>> task_id = TaskCreateTool.static_call(
        ...     prompt="Generate monthly report",
        ...     name="monthly-report"
        ... )
    """

    @staticmethod
    def static_call(
        prompt: str,
        name: str = "",
        subagent_type: str = "general-purpose",
        **context_variables,
    ) -> str:
        """Create a subagent task for background execution.

        Args:
            prompt: Task description for the subagent.
            name: Optional name for the task.
            subagent_type: Type of agent to spawn.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Task ID or error message.
        """
        return AgentTool.static_call(
            prompt=prompt,
            subagent_type=subagent_type,
            name=name,
            wait=False,
        )


class SpawnAgents(AgentBaseFn):
    """Spawn several subagents in parallel. Set ``wait=False`` for true async.

    With ``wait=False``, returns the ids immediately and the main agent stays
    in control — use ``AwaitAgents`` / ``CheckAgentMessages`` / ``PeekAgent``
    to coordinate. With ``wait=True`` (the default for back-compat) blocks
    until every spawned agent terminates.

    Example (fire-and-forget, preferred):
        >>> SpawnAgents.static_call(
        ...     agents=[
        ...         {"name": "lint", "prompt": "Run lint and report violations"},
        ...         {"name": "test", "prompt": "Run the test suite"},
        ...     ],
        ...     wait=False,
        ... )

    Example (sync — only when results are needed before the next step):
        >>> SpawnAgents.static_call(agents=[{"prompt": "..."}], wait=True)
    """

    @staticmethod
    def static_call(
        agents: list[dict[str, str]] | str,
        wait: bool = True,
        timeout: float | None = None,
        **context_variables,
    ) -> str:
        """Spawn one or more subagents in parallel.

        Args:
            agents: List of ``{"prompt": ..., "name"?: ..., "subagent_type"?: ...}``
                dicts. Accepted as a Python list or a JSON-encoded string.
            wait: Block until every agent finishes (default). Set ``False``
                for fire-and-forget — the main agent can then coordinate
                with ``AwaitAgents`` / ``PeekAgent`` / ``CheckAgentMessages``.
            timeout: Maximum seconds to wait when ``wait=True``. Sub-agents
                themselves have no internal timeout; this only bounds the
                main agent's blocking call.
            **context_variables: Forwarded by the dispatcher; unused.

        Returns:
            JSON: when ``wait=False``, an array of task snapshots (ids,
            names, statuses); when ``wait=True``, the same shape plus final
            results.
        """
        from ...agents.definitions import get_agent_definition
        from ...runtime.config_context import get_inheritable

        if isinstance(agents, str):
            agents = _parse_agents_payload(agents)
            if isinstance(agents, str):
                # ``_parse_agents_payload`` returns the original string when
                # all parsing strategies failed, so we can include a precise
                # error rather than the generic JSON one.
                return (
                    "[Error: `agents` must be a list of {prompt, name?, subagent_type?} "
                    "dicts. Got an unparseable string. Pass it as native list/dict, "
                    f"or JSON without extra escaping. Snippet: {agents[:200]}]"
                )
        if not isinstance(agents, list):
            return f"[Error: agents must be a list, got {type(agents).__name__}]"
        if not agents:
            return "[Error: agents list is empty — nothing to spawn]"
        for i, spec in enumerate(agents):
            if not isinstance(spec, dict):
                return f"[Error: agents[{i}] must be a dict, got {type(spec).__name__}]"
            if "prompt" not in spec or not spec["prompt"]:
                return f"[Error: agents[{i}] missing required key `prompt`]"

        mgr = _get_agent_manager()
        mgr.ensure_capacity(len(agents))
        config: dict[str, Any] = get_inheritable()
        tasks = []

        for spec in agents:
            subagent_type = spec.get("subagent_type", "general-purpose")
            agent_def = get_agent_definition(subagent_type)
            eff_config = dict(config)
            if agent_def and agent_def.model:
                eff_config["model"] = agent_def.model

            task = mgr.spawn(
                prompt=spec["prompt"],
                config=eff_config,
                system_prompt=_build_subagent_system_prompt(),
                agent_def=agent_def,
                name=spec.get("name", ""),
            )
            tasks.append(task)

        if not wait:
            return json.dumps([t.snapshot() for t in tasks], indent=2, default=str)

        results = []
        wait_timeout = _spawn_agents_wait_timeout(timeout)
        deadline = None if wait_timeout is None else time.monotonic() + wait_timeout
        for task in tasks:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            mgr.wait(task.id, timeout=remaining)
            results.append(
                {
                    "id": task.id,
                    "name": task.name,
                    "status": task.status,
                    "result": task.result or "",
                    "error": task.error or "",
                    "note": (
                        "Agent is still running; use TaskGetTool or TaskListTool to check it later."
                        if task.status in ("pending", "running")
                        else ""
                    ),
                }
            )
        return json.dumps(results, indent=2, default=str)


class TaskGetTool(AgentBaseFn):
    """Get the current status and result of a task.

    Example:
        >>> TaskGetTool.static_call(task_id="abc123")
    """

    @staticmethod
    def static_call(task_id: str, **context_variables) -> str:
        """Get task status and the latest agent-visible content.

        Args:
            task_id: ID or name of the task to query.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            JSON task status with completed output or compact running progress.
        """
        mgr = _get_agent_manager()
        task = mgr.tasks.get(task_id) or mgr.get_by_name(task_id)
        if not task:
            return f"Error: task '{task_id}' not found."
        payload: dict[str, Any] = {
            "id": task.id,
            "name": task.name,
            "status": task.status,
        }
        if task.status == "completed":
            payload["result"] = task.result or ""
        elif task.status == "failed":
            payload["error"] = task.error or task.result or "Agent failed."
        elif task.status == "cancelled":
            payload["result"] = task.result or "[Sub-agent was cancelled.]"
        else:
            payload["current_tool"] = task.current_tool
            payload["recent_output"] = task.recent_output_text()
            payload["tool_calls_count"] = task.tool_calls_count
            payload["note"] = "Agent is still running; use TaskOutputTool after it completes."
        return json.dumps(payload, indent=2)


class TaskListTool(AgentBaseFn):
    """List all active and recent tasks.

    Example:
        >>> TaskListTool.static_call()
    """

    @staticmethod
    def static_call(**context_variables) -> str:
        """List all tasks.

        Args:
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Formatted list of tasks with status.
        """
        mgr = _get_agent_manager()
        tasks = mgr.list_tasks()
        if not tasks:
            return "No tasks."
        lines = []
        for t in tasks:
            wt = f" [worktree: {t.worktree_branch}]" if t.worktree_branch else ""
            lines.append(f"- {t.name} ({t.id}) [{t.status}]{wt} — {t.prompt[:60]}")
        return "\n".join(lines)


class TaskOutputTool(AgentBaseFn):
    """Get the output result of a completed task.

    Example:
        >>> TaskOutputTool.static_call(task_id="abc123")
    """

    @staticmethod
    def static_call(task_id: str, **context_variables) -> str:
        """Get task output result.

        Args:
            task_id: ID or name of the task.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            The task's result or status message.
        """
        mgr = _get_agent_manager()
        result = mgr.get_result(task_id)
        if result is None:
            task = mgr.get_by_name(task_id)
            if task:
                result = task.result
        return result or f"No output for task '{task_id}' (may still be running)."


class TaskStopTool(AgentBaseFn):
    """Cancel a running task.

    Example:
        >>> TaskStopTool.static_call(task_id="abc123")
    """

    @staticmethod
    def static_call(task_id: str, **context_variables) -> str:
        """Stop a running task.

        Args:
            task_id: ID of the task to stop.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Confirmation or error message.
        """
        mgr = _get_agent_manager()
        ok = mgr.cancel(task_id)
        return f"Task '{task_id}' cancelled." if ok else f"Could not cancel task '{task_id}'."


class TaskUpdateTool(AgentBaseFn):
    """Send an update message to a running task.

    Example:
        >>> TaskUpdateTool.static_call(task_id="abc123", message="New requirements...")
    """

    @staticmethod
    def static_call(
        task_id: str,
        message: str,
        **context_variables,
    ) -> str:
        """Send a message to a running task.

        Args:
            task_id: Target task ID.
            message: Message to send.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Result from SendMessageTool.
        """
        return SendMessageTool.static_call(target=task_id, message=message)


# ---------------------------------------------------------------------------
# Async sub-agent orchestration — the main agent uses these to coordinate
# background sub-agents without blocking on them. The lifecycle is:
#   1. Spawn with ``wait=False`` (AgentTool / TaskCreateTool / SpawnAgents).
#   2. Optionally call ``AwaitAgents`` to sleep until they finish (or until
#      the user wakes us with a steer / new prompt, or until a timeout).
#   3. Use ``CheckAgentMessages`` to drain notifications, or rely on the
#      streaming-loop auto-drain that splices them into the conversation
#      between iterations.
#   4. ``PeekAgent`` shows what a specific sub-agent is doing right now —
#      current tool, recent output, idle time.
#   5. ``ResetAgent`` cancels and re-spawns with the same (or a fresh)
#      prompt; ``TaskStopTool`` cancels without re-spawn.
# ---------------------------------------------------------------------------


class AwaitAgents(AgentBaseFn):
    """Sleep until tracked sub-agents finish, the user wakes us, or timeout.

    The main agent calls this when it has nothing useful to do until its
    children produce results — instead of polling in a loop, this blocks
    on the manager's condition variable so the wake is event-driven.

    Wake conditions (returned as ``wake_reason``):
      - ``"agents_done"``  — sub-agents reached the wake threshold.
      - ``"user_input"``  — a user steer / new prompt landed; we surrender
        so the main agent can read it on its next turn.
      - ``"cancelled"``    — turn-level cancellation requested.
      - ``"timeout"``      — the requested ``timeout_seconds`` elapsed.

    Example:
        >>> AwaitAgents.static_call(timeout_seconds=30, wake_on="any")
    """

    @staticmethod
    def static_call(
        agent_ids: list[str] | str | None = None,
        wake_on: str = "any",
        timeout_seconds: float = 30.0,
        **context_variables,
    ) -> str:
        """Sleep with event-driven wakeups.

        Args:
            agent_ids: Optional list of task IDs or names to watch. ``None``
                or empty ⇒ watch every non-terminal sub-agent.
            wake_on: ``"any"`` to wake when at least one watched agent
                reaches terminal, ``"all"`` to wait for every watched agent,
                ``"none"`` to ignore agent status (pure timeout / user-input
                sleep).
            timeout_seconds: Upper bound on the sleep, in seconds. Set to
                ``0`` for "return immediately with current status".
            **context_variables: Forwarded by the tool dispatcher; unused.

        Returns:
            JSON: ``{"wake_reason": "...", "elapsed_seconds": ..., "agents": [...]}``.
        """
        import json as _json
        import time as _time

        from ...runtime.session_context import get_active_session

        mgr = _get_agent_manager()
        ids_input = agent_ids
        if isinstance(ids_input, str):
            try:
                parsed = _json.loads(ids_input)
                ids_input = parsed if isinstance(parsed, list) else [ids_input]
            except Exception:
                ids_input = [s.strip() for s in ids_input.split(",") if s.strip()]

        watched: list[str] = []
        if ids_input:
            for raw in ids_input:
                tid = mgr._by_name.get(raw, raw)
                if tid in mgr.tasks:
                    watched.append(tid)
        else:
            watched = [tid for tid, t in mgr.tasks.items() if t.status in ("pending", "running")]

        wake_mode = wake_on.strip().lower()
        if wake_mode not in {"any", "all", "none"}:
            wake_mode = "any"

        def _terminal_count() -> int:
            return sum(1 for tid in watched if mgr.tasks[tid].status in ("completed", "failed", "cancelled"))

        def _predicate() -> bool:
            if wake_mode == "none" or not watched:
                return False
            if wake_mode == "all":
                return _terminal_count() == len(watched)
            return _terminal_count() >= 1

        session = get_active_session()

        def _extra_wake() -> bool:
            if session is None:
                return False
            if getattr(session, "cancel_requested", False):
                return True
            steers = getattr(session, "pending_steers", None)
            if steers is not None and not steers.empty():
                return True
            return False

        start = _time.monotonic()
        try:
            timeout = max(0.0, float(timeout_seconds))
        except (TypeError, ValueError):
            timeout = 30.0

        if timeout == 0.0:
            wake_reason = "agents_done" if _predicate() else "timeout"
        else:
            satisfied = mgr.wait_for(_predicate, timeout=timeout, extra_wake=_extra_wake)
            if satisfied:
                wake_reason = "agents_done"
            elif session is not None and getattr(session, "cancel_requested", False):
                wake_reason = "cancelled"
            elif session is not None and getattr(session, "pending_steers", None) and not session.pending_steers.empty():
                wake_reason = "user_input"
            else:
                wake_reason = "timeout"

        elapsed = round(_time.monotonic() - start, 2)
        snapshots = [mgr.tasks[tid].snapshot() for tid in watched if tid in mgr.tasks]
        return _json.dumps(
            {
                "wake_reason": wake_reason,
                "elapsed_seconds": elapsed,
                "wake_on": wake_mode,
                "agents": snapshots,
            },
            indent=2,
            default=str,
        )


class CheckAgentMessages(AgentBaseFn):
    """Drain the mailbox of sub-agent events accumulated since the last call.

    Each call returns events newer than the recorded cursor and advances it,
    so two consecutive calls won't return the same event twice. Use this
    when the main agent wants to peek without sleeping — for passive auto-
    delivery the streaming loop already splices events between iterations.

    Example:
        >>> CheckAgentMessages.static_call()
    """

    @staticmethod
    def static_call(
        since_seq: int = 0,
        peek: bool = False,
        **context_variables,
    ) -> str:
        """Return queued sub-agent events.

        Args:
            since_seq: Return only events with sequence > this value. Defaults
                to ``0`` (everything currently buffered).
            peek: When ``True``, do not consume the buffer — useful for
                observability without disturbing the auto-drain cursor.
            **context_variables: Unused.

        Returns:
            JSON ``{"latest_seq": ..., "events": [...]}``.
        """
        import json as _json

        mgr = _get_agent_manager()
        events = mgr.peek_mailbox(since_seq=since_seq) if peek else mgr.drain_mailbox(since_seq=since_seq)
        return _json.dumps(
            {
                "latest_seq": mgr.latest_seq(),
                "events": events,
            },
            indent=2,
            default=str,
        )


class PeekAgent(AgentBaseFn):
    """Show what a specific sub-agent is doing right now.

    Returns a snapshot with the current tool name, recent output (last
    ~2 KB), idle time, and tool-call count — the information the main
    agent needs to decide "is this agent stuck or working?" without
    cancelling it.

    Example:
        >>> PeekAgent.static_call(target="researcher-1")
    """

    @staticmethod
    def static_call(
        target: str,
        **context_variables,
    ) -> str:
        """Return a rich status snapshot for the named sub-agent."""
        import json as _json

        mgr = _get_agent_manager()
        task = mgr.get_by_name(target) or mgr.tasks.get(target)
        if task is None:
            return f"Error: agent '{target}' not found."
        return _json.dumps(task.snapshot(), indent=2, default=str)


class ResetAgent(AgentBaseFn):
    """Cancel a sub-agent and immediately respawn it with the same spec.

    Useful when an agent is stuck or has gone off-track. Passing
    ``new_prompt`` replaces the original task description; omitting it
    re-runs the original prompt.

    Example:
        >>> ResetAgent.static_call(target="researcher-1", new_prompt="Try again with focus on X")
    """

    @staticmethod
    def static_call(
        target: str,
        new_prompt: str = "",
        **context_variables,
    ) -> str:
        """Cancel + respawn ``target``; return the new task's snapshot or an error."""
        import json as _json

        mgr = _get_agent_manager()
        new_task = mgr.reset(target, new_prompt=new_prompt)
        if new_task is None:
            return f"Error: cannot reset '{target}' (unknown or never spawned via spawn())."
        return _json.dumps(
            {
                "reset_target": target,
                "new_task": new_task.snapshot(),
            },
            indent=2,
            default=str,
        )


class HandoffTool(AgentBaseFn):
    """Hand off a task to another specialized agent.

    Transfers context and control to a more appropriate agent type.

    Example:
        >>> HandoffTool.static_call(
        ...     target_agent="reviewer",
        ...     reason="Requires code review expertise",
        ...     context_summary="Auth module changes"
        ... )
    """

    @staticmethod
    def static_call(
        target_agent: str,
        reason: str,
        context_summary: str = "",
        prompt: str = "",
        timeout: float | None = None,
        **context_variables,
    ) -> str:
        """Hand off to another agent type.

        Args:
            target_agent: Agent type to hand off to.
            reason: Explanation for the handoff.
            context_summary: Summary of current context.
            prompt: Optional specific task for the handoff.
            timeout: Maximum seconds to wait before returning a running snapshot.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Result from the handoff agent.
        """
        from ...agents.definitions import get_agent_definition
        from ...runtime.config_context import emit_event, get_inheritable

        mgr = _get_agent_manager()
        agent_def = get_agent_definition(target_agent)

        config: dict[str, Any] = get_inheritable()
        if agent_def and agent_def.model:
            config["model"] = agent_def.model

        handoff_prompt = f"""## Handoff from parent agent

**Reason:** {reason}

**Context:** {context_summary or "(no context provided)"}

**Your task:** {prompt or "(continue from where the previous agent left off)"}

You are receiving this task via handoff. The previous agent determined that
your specialization ({target_agent}) is better suited for this work.
"""

        emit_event(
            "agent_handoff",
            {
                "from": "parent",
                "to": target_agent,
                "reason": reason,
            },
        )

        task = mgr.spawn(
            prompt=handoff_prompt,
            config=config,
            system_prompt="You are a helpful AI assistant.",
            agent_def=agent_def,
            name=f"handoff-{target_agent}",
        )

        mgr.wait(task.id, timeout=_subagent_wait_timeout(timeout))

        if task.status == "completed" and task.result:
            return task.result
        if task.status == "failed":
            return f"Handoff to {target_agent} failed: {task.error}"
        snapshot = task.snapshot()
        snapshot["note"] = "Handoff agent is still running; use TaskGetTool or TaskListTool to check it later."
        return json.dumps(snapshot, indent=2, default=str)
