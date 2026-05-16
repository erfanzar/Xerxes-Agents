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
"""Worktree-isolated subagent spawning, lifecycle, and cancellation.

A :class:`SubAgentManager` runs subagent prompts on a thread pool
(:class:`~concurrent.futures.ThreadPoolExecutor`) and tracks each one as a
:class:`SubAgentTask`. Tasks transition through ``pending → running →
{completed, failed, cancelled}`` and may run in an isolated git worktree
(branch ``xerxes-agent-<hex>``) when ``isolation == "worktree"``.

Cancellation is cooperative: ``cancel()`` flips ``_cancel_flag`` and marks
the task ``cancelled``; the streaming loop polls the flag between steps and
exits at the next safe boundary. A task already in a terminal state cannot
be cancelled. Worktrees are removed automatically only when they have no
uncommitted changes or unpushed commits — anything else is left for the user
to inspect.

The per-task ``_inbox`` queue lets a parent send follow-up prompts to a
running subagent; messages enqueued after the first run are picked up in
order, each driving another streaming pass with the same effective config.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import tempfile
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .definitions import AgentDefinition

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_SUBAGENT_BLOCKED_TOOLS = {
    "AgentTool",
    "SpawnAgents",
    "TaskCreateTool",
    "SkillTool",
}

_SUBAGENT_CALLER_PROMPT = (
    "You are now running as a subagent. All the `user` messages are sent by the main agent. "
    "The main agent cannot see your context, it can only see your last message when you finish the task. "
    "You must treat the parent agent as your caller. Do not directly ask the end user questions. "
    "If something is unclear, explain the ambiguity in your final summary to the parent agent."
)


@dataclass
class SubAgentTask:
    """One spawned subagent and the state needed to observe and cancel it.

    The terminal states are ``completed``, ``failed`` and ``cancelled``; once
    ``_done_event`` is set the task will not transition again. ``_cancel_flag``
    is the cooperative cancellation signal polled by the streaming loop.

    Attributes:
        id: Unique 12-char hex task id.
        prompt: Initial prompt sent to the subagent.
        status: One of ``pending``, ``running``, ``completed``, ``failed``,
            ``cancelled``.
        result: Final assistant text (truncated to 500 chars when emitted in
            events).
        depth: Delegation depth — refused if it would exceed
            :attr:`SubAgentManager.max_depth`.
        name: Short human label used for lookups via
            :meth:`SubAgentManager.get_by_name`.
        agent_def_name: Backing :class:`AgentDefinition` name, if any.
        worktree_path: Filesystem path of the isolated git worktree, when used.
        worktree_branch: Branch name created for the worktree.
        error: Failure message when ``status == "failed"``.
        messages_sent: Count of follow-ups delivered via ``send_message``.
    """

    id: str = ""
    prompt: str = ""
    status: str = "pending"
    result: str | None = None
    depth: int = 0
    name: str = ""
    agent_def_name: str = ""
    worktree_path: str = ""
    worktree_branch: str = ""
    error: str = ""
    messages_sent: int = 0
    _cancel_flag: bool = field(default=False, repr=False)
    _future: Future | None = field(default=None, repr=False)
    _inbox: queue.Queue = field(default_factory=queue.Queue, repr=False)
    _done_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe dict view of the task.

        The prompt is truncated to 200 chars and the result to 500 chars so
        the snapshot is cheap to log and ship over the bridge. Internal
        synchronisation primitives (future, event, cancel flag) are omitted.
        """
        return {
            "id": self.id,
            "name": self.name,
            "prompt": self.prompt[:200],
            "status": self.status,
            "result": self.result[:500] if self.result else None,
            "depth": self.depth,
            "agent_def": self.agent_def_name,
            "worktree_path": self.worktree_path,
            "worktree_branch": self.worktree_branch,
            "error": self.error,
            "messages_sent": self.messages_sent,
            "inbox_size": self._inbox.qsize(),
        }


def _git_root(cwd: str) -> str | None:
    """Return the git toplevel of ``cwd``, or ``None`` if not a repo."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return r.stdout.strip()
    except Exception:
        return None


def _create_worktree(base_dir: str) -> tuple[str, str]:
    """Create a fresh git worktree on a new ``xerxes-agent-*`` branch.

    Returns ``(worktree_path, branch_name)``. Raises
    :class:`subprocess.CalledProcessError` if ``git worktree add`` fails.
    """
    branch = f"xerxes-agent-{uuid.uuid4().hex[:8]}"
    wt_path = tempfile.mkdtemp(prefix="xerxes-agent-wt-")
    os.rmdir(wt_path)
    subprocess.run(
        ["git", "worktree", "add", "-b", branch, wt_path],
        cwd=base_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return wt_path, branch


def _remove_worktree(wt_path: str, branch: str, base_dir: str) -> None:
    """Force-remove ``wt_path`` and delete its branch, swallowing errors."""
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", wt_path],
            cwd=base_dir,
            capture_output=True,
        )
    except Exception:
        pass
    try:
        subprocess.run(
            ["git", "branch", "-D", branch],
            cwd=base_dir,
            capture_output=True,
        )
    except Exception:
        pass


def _has_worktree_changes(wt_path: str) -> bool:
    """Return ``True`` if ``wt_path`` has uncommitted edits or unpushed commits.

    Used to decide whether a worktree is safe to auto-remove after a subagent
    finishes; anything dirty is left in place for the user to inspect.
    """
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=wt_path,
            capture_output=True,
            text=True,
        )
        if r.stdout.strip():
            return True
        r2 = subprocess.run(
            ["git", "log", "--oneline", "HEAD", "--not", "--remotes", "-1"],
            cwd=wt_path,
            capture_output=True,
            text=True,
        )
        return bool(r2.stdout.strip())
    except Exception:
        return False


class SubAgentManager:
    """Owns the thread pool and registry for all live subagent tasks.

    The manager is a process-singleton in practice: the streaming loop hands
    every spawn through it so cancellation, name lookup, and inbox delivery
    work uniformly. ``tasks`` is the canonical registry; ``_by_name`` is the
    secondary lookup populated when callers supply a name on ``spawn``.
    """

    def __init__(self, max_concurrent: int = 8, max_depth: int = 5):
        """Build a manager backed by a thread pool of ``max_concurrent``.

        Args:
            max_concurrent: Initial worker count for the thread pool.
            max_depth: Hard cap on delegation depth. Spawns at or beyond this
                depth fail immediately with status ``failed``.
        """
        self.tasks: dict[str, SubAgentTask] = {}
        self._by_name: dict[str, str] = {}
        self.max_concurrent = max_concurrent
        self.max_depth = max_depth
        self._pool = ThreadPoolExecutor(max_workers=max_concurrent)
        self._agent_runner: Any = None
        self._tool_executor: Any = None
        self._tool_schemas: list[dict[str, Any]] | None = None

    def ensure_capacity(self, min_concurrent: int) -> bool:
        """Grow the pool to at least ``min_concurrent`` workers.

        :class:`ThreadPoolExecutor` cannot resize, so this swaps in a fresh
        executor for future submissions. Tasks already running in the old
        executor keep running on their original threads.
        """
        if min_concurrent <= self.max_concurrent:
            return True
        self._pool.shutdown(wait=False, cancel_futures=False)
        self.max_concurrent = min_concurrent
        self._pool = ThreadPoolExecutor(max_workers=min_concurrent)
        return True

    def set_runner(self, runner: Any) -> None:
        """Install the callable that drives a subagent for one prompt.

        When unset, the manager falls back to running the in-process
        streaming loop directly via :func:`_run_streaming_loop`. Setting a
        custom runner lets tests stub out execution.
        """
        self._agent_runner = runner

    def spawn(
        self,
        prompt: str,
        config: dict[str, Any],
        system_prompt: str,
        depth: int = 0,
        agent_def: AgentDefinition | None = None,
        isolation: str = "",
        name: str = "",
    ) -> SubAgentTask:
        """Spawn a subagent and submit it to the thread pool.

        The task is registered before submission so callers see it via
        :meth:`list_tasks` even if the future has not started yet. Returns
        immediately; use :meth:`wait` to block on completion.

        If ``depth >= max_depth`` or worktree creation fails, the task is
        returned already in ``failed`` state (no work is scheduled) and a
        descriptive ``error`` / ``result`` are recorded.

        Args:
            prompt: Initial subagent prompt. Worktree instructions are
                appended automatically when isolation is enabled.
            config: Runtime config copied into the task; ``_tools_*`` keys
                are derived from ``agent_def`` if provided.
            system_prompt: Base system prompt prepended to the subagent's
                conversation. ``agent_def.system_prompt`` is layered on top
                with the caller-context preamble.
            depth: Current delegation depth; passed through ``depth+1`` to
                the inner loop and checked against ``max_depth``.
            agent_def: Optional :class:`AgentDefinition` whose model, system
                prompt, tool allow/deny lists, and isolation override the
                defaults.
            isolation: ``"worktree"`` to run in an isolated branch; empty
                falls back to ``agent_def.isolation``.
            name: Optional short name registered for
                :meth:`get_by_name` / :meth:`send_message` lookups.
        """
        task_id = uuid.uuid4().hex[:12]
        short_name = name or task_id[:8]
        task = SubAgentTask(
            id=task_id,
            prompt=prompt,
            depth=depth,
            name=short_name,
            agent_def_name=agent_def.name if agent_def else "",
        )
        self.tasks[task_id] = task
        if name:
            self._by_name[name] = task_id

        if depth >= self.max_depth:
            task.status = "failed"
            task.error = f"Max depth ({self.max_depth}) exceeded"
            task.result = task.error
            task._done_event.set()
            return task

        eff_config = dict(config)
        eff_system = system_prompt

        if agent_def:
            if agent_def.model:
                eff_config["model"] = agent_def.model
            if agent_def.system_prompt:
                eff_system = _SUBAGENT_CALLER_PROMPT + "\n\n" + agent_def.system_prompt.rstrip() + "\n\n" + system_prompt
            if not isolation and agent_def.isolation:
                isolation = agent_def.isolation

            eff_config["_tools_allowed"] = agent_def.allowed_tools
            eff_config["_tools_excluded"] = agent_def.exclude_tools
            if agent_def.tools:
                eff_config["_tools_whitelist"] = agent_def.tools

        worktree_path = ""
        worktree_branch = ""
        base_dir = os.getcwd()

        if isolation == "worktree":
            git_root = _git_root(base_dir)
            if not git_root:
                task.status = "failed"
                task.error = "isolation='worktree' requires a git repository"
                task.result = task.error
                task._done_event.set()
                return task
            try:
                worktree_path, worktree_branch = _create_worktree(git_root)
                task.worktree_path = worktree_path
                task.worktree_branch = worktree_branch
                prompt += (
                    f"\n\n[Note: You are working in an isolated git worktree at "
                    f"{worktree_path} (branch: {worktree_branch}). "
                    f"Your changes are isolated from the main workspace at {git_root}. "
                    f"Commit your changes before finishing so they can be reviewed/merged.]"
                )
            except Exception as e:
                task.status = "failed"
                task.error = f"Failed to create worktree: {e}"
                task.result = task.error
                task._done_event.set()
                return task

        runner = self._agent_runner

        def _run() -> None:
            """Execute the task body, drain the inbox, and finalise state."""
            from xerxes.runtime.config_context import emit_event

            task.status = "running"
            emit_event(
                "agent_spawn",
                {
                    "task_id": task.id,
                    "agent_name": task.name,
                    "agent_type": task.agent_def_name,
                    "prompt": task.prompt[:200],
                    "depth": task.depth,
                    "isolation": isolation,
                },
            )
            old_cwd = os.getcwd()
            try:
                if worktree_path:
                    os.chdir(worktree_path)

                if runner:
                    result = runner(
                        prompt,
                        eff_config,
                        eff_system,
                        depth + 1,
                        lambda: task._cancel_flag,
                    )
                    task.result = result
                else:
                    task.result = _run_streaming_loop(
                        prompt,
                        eff_config,
                        eff_system,
                        depth + 1,
                        task,
                        tool_executor=self._tool_executor,
                        tool_schemas=self._tool_schemas,
                    )

                if task._cancel_flag:
                    task.status = "cancelled"
                else:
                    task.status = "completed"

                while not task._inbox.empty() and not task._cancel_flag:
                    inbox_msg = task._inbox.get_nowait()
                    task.status = "running"
                    if runner:
                        result = runner(
                            inbox_msg,
                            eff_config,
                            eff_system,
                            depth + 1,
                            lambda: task._cancel_flag,
                        )
                        task.result = result
                    else:
                        task.result = _run_streaming_loop(
                            inbox_msg,
                            eff_config,
                            eff_system,
                            depth + 1,
                            task,
                            tool_executor=self._tool_executor,
                            tool_schemas=self._tool_schemas,
                        )

                    if not task._cancel_flag:
                        task.status = "completed"

            except Exception as e:
                task.status = "failed"
                task.error = str(e)
                task.result = f"Error: {e}"
                logger.error("Sub-agent %s failed: %s", task_id, e)
            finally:
                if task.status in _TERMINAL_STATUSES:
                    task._done_event.set()
                emit_event(
                    "agent_done",
                    {
                        "task_id": task.id,
                        "agent_name": task.name,
                        "agent_type": task.agent_def_name,
                        "status": task.status,
                        "result": (task.result or "")[:500],
                    },
                )
                if worktree_path:
                    os.chdir(old_cwd)
                    if not _has_worktree_changes(worktree_path):
                        _remove_worktree(worktree_path, worktree_branch, old_cwd)

        task._future = self._pool.submit(_run)
        return task

    def wait(self, task_id: str, timeout: float | None = None) -> SubAgentTask | None:
        """Block until ``task_id`` reaches a terminal state or ``timeout``.

        Returns the task object whether or not the wait timed out — callers
        should check ``status`` to distinguish. ``None`` is returned only
        when ``task_id`` is unknown.
        """
        task = self.tasks.get(task_id)
        if task is None:
            return None
        if task._done_event.is_set():
            return task
        task._done_event.wait(timeout=timeout)
        return task

    def wait_all(
        self,
        task_ids: list[str] | None = None,
        timeout: float | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Wait on a batch of tasks and split snapshots by terminal state.

        ``task_ids`` defaults to every pending or running task. The returned
        dict has two keys, ``"completed"`` and ``"pending"`` — tasks that
        timed out before reaching a terminal state land in ``pending``.
        """
        ids = task_ids or [tid for tid, t in self.tasks.items() if t.status in ("pending", "running")]
        completed = []
        pending = []
        for tid in ids:
            task = self.wait(tid, timeout=timeout)
            if task is None:
                continue
            if task.status in ("completed", "failed", "cancelled"):
                completed.append(task.snapshot())
            else:
                pending.append(task.snapshot())
        return {"completed": completed, "pending": pending}

    def send_message(self, task_id_or_name: str, message: str) -> bool:
        """Enqueue ``message`` for a running or pending subagent.

        The inbox is drained after the current run completes; once a task
        has reached a terminal state, follow-ups are rejected and ``False``
        is returned. ``task_id_or_name`` may be either the raw task id or
        the name supplied to :meth:`spawn`.
        """
        task_id = self._by_name.get(task_id_or_name, task_id_or_name)
        task = self.tasks.get(task_id)
        if task is None:
            return False
        if task.status not in ("running", "pending"):
            return False
        task._inbox.put(message)
        task.messages_sent += 1
        return True

    def get_result(self, task_id: str) -> str | None:
        """Return the current ``result`` string for ``task_id`` (may be partial)."""
        task = self.tasks.get(task_id)
        return task.result if task else None

    def cancel(self, task_id: str) -> bool:
        """Cooperatively cancel a pending or running task.

        Sets ``_cancel_flag`` (polled by the streaming loop), transitions the
        task to ``cancelled``, and releases :meth:`wait` callers. Tasks
        already in a terminal state are not touched and ``False`` is
        returned.
        """
        task = self.tasks.get(task_id)
        if task is None:
            return False
        if task.status in ("running", "pending"):
            task._cancel_flag = True
            task.status = "cancelled"
            task.result = task.result or "[Sub-agent was cancelled.]"
            task._done_event.set()
            return True
        return False

    def cancel_all(self) -> int:
        """Cancel every non-terminal task; return how many were transitioned."""
        n = 0
        for task in self.tasks.values():
            if task.status in ("running", "pending"):
                task._cancel_flag = True
                task.status = "cancelled"
                task.result = task.result or "[Sub-agent was cancelled.]"
                task._done_event.set()
                n += 1
        return n

    def list_tasks(self) -> list[SubAgentTask]:
        """Return every tracked :class:`SubAgentTask`."""
        return list(self.tasks.values())

    def list_snapshots(self) -> list[dict[str, Any]]:
        """Return :meth:`SubAgentTask.snapshot` for every tracked task."""
        return [t.snapshot() for t in self.tasks.values()]

    def get_by_name(self, name: str) -> SubAgentTask | None:
        """Return the task registered under ``name`` (from :meth:`spawn`)."""
        task_id = self._by_name.get(name)
        return self.tasks.get(task_id) if task_id else None

    def shutdown(self) -> None:
        """Signal cancellation on every active task and join the pool.

        Sets ``_cancel_flag`` on running and pending tasks so the streaming
        loop exits at the next checkpoint, then waits for the executor to
        drain. Subagents that ignore the flag will still keep the pool open
        until they finish naturally.
        """
        for task in self.tasks.values():
            if task.status in ("running", "pending"):
                task._cancel_flag = True
        self._pool.shutdown(wait=True)

    def summary(self) -> str:
        """Return a Markdown overview suitable for the ``/agents`` view."""
        lines = [
            "# Sub-Agent Tasks",
            "",
            f"Total: {len(self.tasks)}",
            f"Running: {sum(1 for t in self.tasks.values() if t.status == 'running')}",
            f"Completed: {sum(1 for t in self.tasks.values() if t.status == 'completed')}",
            "",
        ]
        for task in self.tasks.values():
            wt = f" [worktree: {task.worktree_branch}]" if task.worktree_branch else ""
            agent = f" ({task.agent_def_name})" if task.agent_def_name else ""
            lines.append(f"- **{task.name}**{agent} [{task.status}]{wt} — {task.prompt[:60]}")
        return "\n".join(lines)


def _run_streaming_loop(
    prompt: str,
    config: dict[str, Any],
    system_prompt: str,
    depth: int,
    task: SubAgentTask,
    tool_executor: Any = None,
    tool_schemas: list[dict[str, Any]] | None = None,
) -> str:
    """Drive :func:`xerxes.streaming.loop.run` for one subagent prompt.

    Tool schemas and executor are filtered via :func:`_filter_subagent_tools`
    so a subagent never inherits orchestration tools that would let it spawn
    another swarm. Streaming events are re-emitted as ``agent_*`` events on
    the bridge so the TUI can render them. Returns the concatenated text
    output, which the caller assigns to ``task.result``.
    """
    from xerxes.runtime.config_context import emit_event
    from xerxes.streaming.events import AgentState, TextChunk, ThinkingChunk, ToolEnd, ToolStart
    from xerxes.streaming.loop import run

    state = AgentState()
    output_parts: list[str] = []
    eff_tool_schemas, eff_tool_executor = _filter_subagent_tools(
        tool_schemas=tool_schemas,
        tool_executor=tool_executor,
        config=config,
        is_subagent=depth > 0,
    )

    for event in run(
        user_message=prompt,
        state=state,
        config=config,
        system_prompt=system_prompt,
        tool_executor=eff_tool_executor,
        tool_schemas=eff_tool_schemas,
        depth=depth,
        cancel_check=lambda: task._cancel_flag,
    ):
        if isinstance(event, TextChunk):
            output_parts.append(event.text)
            emit_event(
                "agent_text",
                {
                    "task_id": task.id,
                    "agent_name": task.name,
                    "agent_type": task.agent_def_name,
                    "text": event.text,
                },
            )
        elif isinstance(event, ThinkingChunk):
            emit_event(
                "agent_thinking",
                {
                    "task_id": task.id,
                    "agent_name": task.name,
                    "text": event.text,
                },
            )
        elif isinstance(event, ToolStart):
            emit_event(
                "agent_tool_start",
                {
                    "task_id": task.id,
                    "agent_name": task.name,
                    "agent_type": task.agent_def_name,
                    "tool_call_id": event.tool_call_id,
                    "tool_name": event.name,
                    "inputs": event.inputs,
                },
            )
        elif isinstance(event, ToolEnd):
            emit_event(
                "agent_tool_end",
                {
                    "task_id": task.id,
                    "agent_name": task.name,
                    "agent_type": task.agent_def_name,
                    "tool_call_id": event.tool_call_id,
                    "tool_name": event.name,
                    "result": event.result[:500] if len(event.result) > 500 else event.result,
                    "permitted": event.permitted,
                    "duration_ms": event.duration_ms,
                },
            )

    return "".join(output_parts)


def _filter_subagent_tools(
    *,
    tool_schemas: list[dict[str, Any]] | None,
    tool_executor: Any,
    config: dict[str, Any],
    is_subagent: bool,
) -> tuple[list[dict[str, Any]] | None, Any]:
    """Apply per-agent tool allow/deny lists and block recursive delegation.

    Reads three keys out of ``config`` populated by :meth:`SubAgentManager.spawn`:
    ``_tools_whitelist``, ``_tools_allowed``, ``_tools_excluded``. When
    ``is_subagent`` is true, orchestration tools listed in
    :data:`_SUBAGENT_BLOCKED_TOOLS` are removed unless
    ``_allow_subagent_delegation`` is set, preventing a subagent from
    re-triggering Skill, SpawnAgents, etc.
    """
    if tool_schemas is None:
        return None, tool_executor

    all_names = {str(schema.get("name", "")) for schema in tool_schemas if schema.get("name")}
    whitelist = config.get("_tools_whitelist")
    allowed_tools = config.get("_tools_allowed")
    excluded_tools = set(config.get("_tools_excluded") or [])

    allowed = set(whitelist) if whitelist else all_names.copy()
    if allowed_tools:
        allowed &= set(allowed_tools)
    allowed -= excluded_tools

    if is_subagent and not config.get("_allow_subagent_delegation"):
        allowed -= _SUBAGENT_BLOCKED_TOOLS

    filtered_schemas = [schema for schema in tool_schemas if schema.get("name", "") in allowed]
    if allowed == all_names or tool_executor is None:
        return filtered_schemas, tool_executor

    def _filtered_executor(tool_name: str, tool_input: dict[str, Any]) -> str:
        if tool_name not in allowed:
            return f"Error: tool '{tool_name}' is not allowed for this agent."
        return tool_executor(tool_name, tool_input)

    return filtered_schemas, _filtered_executor


__all__ = [
    "SubAgentManager",
    "SubAgentTask",
]
