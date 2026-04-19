# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.


"""Thread-pool-based sub-agent system with git worktree isolation.

Inspired by the nano-claude-code ``SubAgentManager``, this module provides:

- **SubAgentTask**: Lifecycle-tracked task with inbox queue, cancel flag,
  and git worktree metadata.
- **SubAgentManager**: Thread-pool manager that spawns sub-agents, supports
  named agents, ``SendMessage`` inbox, git worktree isolation, and
  max-depth recursion limits.

Key differences from the async ``SpawnedAgentManager`` in
``xerxes.operators.subagents``:

- Uses ``ThreadPoolExecutor`` instead of ``asyncio.Task`` — works from
  both sync and async contexts.
- Supports git worktree isolation (``isolation="worktree"``).
- Supports named agents addressable via ``send_message(name, text)``.
- Each task has an inbox queue for follow-up messages.
- Integrates with ``AgentDefinition`` for typed agent specializations.

Usage::

    from xerxes.agents.subagent_manager import SubAgentManager
    from xerxes.agents.definitions import get_agent_definition

    mgr = SubAgentManager(max_concurrent=5)

    task = mgr.spawn(
        prompt="Review this PR for security issues",
        config={"model": "gpt-4o"},
        system_prompt="You are a coding assistant.",
        agent_def=get_agent_definition("reviewer"),
        name="security-review",
    )


    mgr.send_message("security-review", "Also check for SQL injection.")


    mgr.wait(task.id, timeout=60)
    print(task.result)
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import tempfile
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .definitions import AgentDefinition

logger = logging.getLogger(__name__)


@dataclass
class SubAgentTask:
    """Represents a sub-agent task with lifecycle tracking.

    Attributes:
        id: Unique task identifier.
        prompt: The initial prompt for this task.
        status: Current status (``"pending"``, ``"running"``, ``"completed"``,
            ``"failed"``, ``"cancelled"``).
        result: The final output text (set on completion).
        depth: Nesting depth (0 = top-level, 1 = first sub-agent, etc.).
        name: Optional human-readable name for addressing via ``send_message``.
        agent_def_name: Name of the ``AgentDefinition`` used (if any).
        worktree_path: Path to the git worktree (if ``isolation="worktree"``).
        worktree_branch: Branch name of the worktree.
        error: Error message if the task failed.
        messages_sent: Count of follow-up messages sent to this task.
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

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot of the task state."""
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
    """Return the git root directory for cwd, or None."""
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
    """Create a temporary git worktree. Returns (path, branch_name)."""
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
    """Remove a git worktree and delete its branch (best-effort)."""
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
    """Check if a worktree has uncommitted or committed changes."""
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
    """Thread-pool-based manager for concurrent sub-agent tasks.

    Supports spawning agents with typed definitions, git worktree
    isolation, named addressing, inbox message queues, and max-depth
    recursion limits.
    """

    def __init__(self, max_concurrent: int = 5, max_depth: int = 5):
        """Initialize the manager.

        Args:
            max_concurrent: Maximum number of concurrent sub-agents.
            max_depth: Maximum nesting depth to prevent infinite recursion.
        """
        self.tasks: dict[str, SubAgentTask] = {}
        self._by_name: dict[str, str] = {}
        self.max_concurrent = max_concurrent
        self.max_depth = max_depth
        self._pool = ThreadPoolExecutor(max_workers=max_concurrent)
        self._agent_runner: Any = None
        self._tool_executor: Any = None
        self._tool_schemas: list[dict[str, Any]] | None = None

    def set_runner(self, runner: Any) -> None:
        """Set the agent runner callable.

        The runner should have signature::

            runner(prompt, config, system_prompt, depth, cancel_check) -> str

        where it runs the full agent loop and returns the final response text.
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
        """Spawn a new sub-agent task.

        Args:
            prompt: User message for the sub-agent.
            config: Agent configuration dict (copied before modification).
            system_prompt: Base system prompt.
            depth: Current nesting depth.
            agent_def: Optional typed agent definition with overrides.
            isolation: ``""`` for normal, ``"worktree"`` for git worktree.
            name: Optional human-readable name for ``send_message`` addressing.

        Returns:
            A :class:`SubAgentTask` tracking the spawned work.
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
            return task

        eff_config = dict(config)
        eff_system = system_prompt

        if agent_def:
            if agent_def.model:
                eff_config["model"] = agent_def.model
            if agent_def.system_prompt:
                eff_system = agent_def.system_prompt.rstrip() + "\n\n" + system_prompt
            if not isolation and agent_def.isolation:
                isolation = agent_def.isolation
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
                return task

        runner = self._agent_runner

        def _run() -> None:
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
        """Block until a task completes or timeout expires.

        Args:
            task_id: Task ID to wait for.
            timeout: Maximum wait time in seconds. ``None`` = wait forever.

        Returns:
            The task, or ``None`` if task_id is unknown.
        """
        task = self.tasks.get(task_id)
        if task is None:
            return None
        if task._future is not None:
            try:
                task._future.result(timeout=timeout)
            except Exception:
                pass
        return task

    def wait_all(
        self,
        task_ids: list[str] | None = None,
        timeout: float | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Wait for multiple tasks.

        Args:
            task_ids: Tasks to wait for. ``None`` = all running tasks.
            timeout: Maximum wait time in seconds per task.

        Returns:
            Dict with ``"completed"`` and ``"pending"`` task snapshots.
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
        """Send a follow-up message to a running sub-agent.

        The message is queued and processed after the current work completes.

        Args:
            task_id_or_name: Task ID or the human-readable name.
            message: Message text.

        Returns:
            ``True`` if queued, ``False`` if task not found or already done.
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
        """Return the result string for a completed task, or None."""
        task = self.tasks.get(task_id)
        return task.result if task else None

    def cancel(self, task_id: str) -> bool:
        """Request cancellation of a running task.

        Returns:
            ``True`` if the cancel flag was set, ``False`` otherwise.
        """
        task = self.tasks.get(task_id)
        if task is None:
            return False
        if task.status in ("running", "pending"):
            task._cancel_flag = True
            return True
        return False

    def list_tasks(self) -> list[SubAgentTask]:
        """Return all tracked tasks."""
        return list(self.tasks.values())

    def list_snapshots(self) -> list[dict[str, Any]]:
        """Return serializable snapshots of all tasks."""
        return [t.snapshot() for t in self.tasks.values()]

    def get_by_name(self, name: str) -> SubAgentTask | None:
        """Look up a task by its human-readable name."""
        task_id = self._by_name.get(name)
        return self.tasks.get(task_id) if task_id else None

    def shutdown(self) -> None:
        """Cancel all running tasks and shut down the thread pool."""
        for task in self.tasks.values():
            if task.status in ("running", "pending"):
                task._cancel_flag = True
        self._pool.shutdown(wait=True)

    def summary(self) -> str:
        """Return a markdown summary of all tasks."""
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
    """Run the streaming agent loop, emitting events for parent visibility."""
    from xerxes.runtime.config_context import emit_event
    from xerxes.streaming.events import AgentState, TextChunk, ThinkingChunk, ToolEnd, ToolStart
    from xerxes.streaming.loop import run

    state = AgentState()
    output_parts: list[str] = []

    for event in run(
        user_message=prompt,
        state=state,
        config=config,
        system_prompt=system_prompt,
        tool_executor=tool_executor,
        tool_schemas=tool_schemas,
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
                    "tool_name": event.name,
                    "result": event.result[:500] if len(event.result) > 500 else event.result,
                    "permitted": event.permitted,
                    "duration_ms": event.duration_ms,
                },
            )

    return "".join(output_parts)


__all__ = [
    "SubAgentManager",
    "SubAgentTask",
]
