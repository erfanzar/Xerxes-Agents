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
"""Sub-agent task manager with git worktree isolation support.

This module provides :class:`SubAgentManager` and :class:`SubAgentTask`, which
enable spawning, monitoring, and cancelling sub-agent tasks in a thread pool.
Tasks may optionally run in isolated git worktrees.
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


@dataclass
class SubAgentTask:
    """Represents a spawned sub-agent task and its lifecycle state.

    Attributes:
        id (str): Unique task identifier.
        prompt (str): The prompt sent to the sub-agent.
        status (str): Current status (e.g., ``"pending"``, ``"running"``, ``"completed"``).
        result (str | None): Final result string, if any.
        depth (int): Delegation depth level.
        name (str): Human-readable short name.
        agent_def_name (str): Name of the agent definition used.
        worktree_path (str): Git worktree path, if isolation was used.
        worktree_branch (str): Git worktree branch name, if used.
        error (str): Error message, if the task failed.
        messages_sent (int): Count of messages sent to the task inbox.
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
        """Return a serializable snapshot of the task state.

        Returns:
            dict[str, Any]: OUT: Summarized fields including prompt preview,
                result preview, and inbox size.
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
    """Determine the git repository root for a directory.

    Args:
        cwd (str): IN: Working directory to query. OUT: Passed to ``git``.

    Returns:
        str | None: OUT: Absolute path to the repository root, or ``None``.
    """
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
    """Create a new git worktree for isolated sub-agent execution.

    Args:
        base_dir (str): IN: Base git repository directory. OUT: Used as the
            working directory for ``git worktree add``.

    Returns:
        tuple[str, str]: OUT: ``(worktree_path, branch_name)``.
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
    """Remove a git worktree and its associated branch.

    Args:
        wt_path (str): IN: Path to the worktree directory. OUT: Passed to
            ``git worktree remove``.
        branch (str): IN: Branch name to delete. OUT: Passed to ``git branch -D``.
        base_dir (str): IN: Base repository directory. OUT: Used as cwd for git.
    """
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
    """Check whether a worktree has uncommitted changes or commits.

    Args:
        wt_path (str): IN: Path to the worktree. OUT: Queried via ``git status``
            and ``git log``.

    Returns:
        bool: OUT: ``True`` if there are uncommitted changes or unpushed commits.
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
    """Manages the lifecycle of sub-agent tasks, including spawning and cancellation.

    Tasks run in a :class:`~concurrent.futures.ThreadPoolExecutor` and may
    optionally be isolated in git worktrees.
    """

    def __init__(self, max_concurrent: int = 8, max_depth: int = 5):
        """Initialize the sub-agent manager.

        Args:
            max_concurrent (int): IN: Maximum number of concurrent tasks. OUT:
                Sets the thread pool worker count.
            max_depth (int): IN: Maximum sub-agent delegation depth. OUT: Used
                to block spawns that would exceed this depth.
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
        """Increase worker capacity before a new batch is submitted.

        ``ThreadPoolExecutor`` cannot resize in place, so this swaps in a new
        executor for future submissions. Existing and queued futures in the old
        executor are left to finish.

        Args:
            min_concurrent (int): IN: Minimum worker count needed. OUT: Used to
                decide whether to replace the executor.

        Returns:
            bool: OUT: ``True`` if capacity is now at least ``min_concurrent``.
        """
        if min_concurrent <= self.max_concurrent:
            return True
        self._pool.shutdown(wait=False, cancel_futures=False)
        self.max_concurrent = min_concurrent
        self._pool = ThreadPoolExecutor(max_workers=min_concurrent)
        return True

    def set_runner(self, runner: Any) -> None:
        """Set the agent runner callable used to execute sub-agent prompts.

        Args:
            runner (Any): IN: Callable that runs a sub-agent. OUT: Stored for
                use during task execution.
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
            prompt (str): IN: The user prompt for the sub-agent. OUT: May be
                appended with worktree instructions if isolation is enabled.
            config (dict[str, Any]): IN: Runtime configuration dict. OUT: May be
                mutated with agent-specific overrides.
            system_prompt (str): IN: Base system prompt. OUT: May be prefixed with
                the agent definition's system prompt.
            depth (int): IN: Current delegation depth. OUT: Incremented for the
                spawned task and checked against ``max_depth``.
            agent_def (AgentDefinition | None): IN: Optional agent definition. OUT:
                Used to override model, system prompt, tools, and isolation.
            isolation (str): IN: Isolation mode (e.g., ``"worktree"``). OUT:
                Defaults to the agent definition's isolation if empty.
            name (str): IN: Optional human-readable task name. OUT: Stored and
                used for name-based lookups.

        Returns:
            SubAgentTask: OUT: The spawned task object (may already be marked
                failed if depth or worktree creation failed).
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
                eff_system = agent_def.system_prompt.rstrip() + "\n\n" + system_prompt
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
            """Internal helper to run."""
            from xerxes.runtime.config_context import emit_event

            """Internal helper to run.
            """
            """Internal helper to run.
            """

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
        """Wait for a task to complete.

        Args:
            task_id (str): IN: Task identifier. OUT: Looked up in the task registry.
            timeout (float | None): IN: Maximum seconds to wait. OUT: Passed to
                the task completion event.

        Returns:
            SubAgentTask | None: OUT: The task object, or ``None`` if not found.
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
        """Wait for multiple tasks and return their snapshots.

        Args:
            task_ids (list[str] | None): IN: Specific IDs to wait for. OUT: Defaults
                to all pending/running tasks if ``None``.
            timeout (float | None): IN: Per-task wait timeout. OUT: Passed to
                :meth:`wait`.

        Returns:
            dict[str, list[dict[str, Any]]]: OUT: Mapping with ``"completed"`` and
                ``"pending"`` snapshot lists.
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
        """Send a follow-up message to a running or pending task.

        Args:
            task_id_or_name (str): IN: Task ID or registered name. OUT: Resolved
                to a task ID.
            message (str): IN: Message text to enqueue. OUT: Put into the task inbox.

        Returns:
            bool: OUT: ``True`` if the message was delivered.
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
        """Get the result of a task.

        Args:
            task_id (str): IN: Task identifier. OUT: Looked up in the registry.

        Returns:
            str | None: OUT: The task result, or ``None`` if not found.
        """
        task = self.tasks.get(task_id)
        return task.result if task else None

    def cancel(self, task_id: str) -> bool:
        """Cancel a single running or pending task.

        Args:
            task_id (str): IN: Task identifier. OUT: Used to set the cancel flag.

        Returns:
            bool: OUT: ``True`` if the task was found and eligible for cancellation.
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
        """Cancel all running and pending tasks.

        Returns:
            int: OUT: Number of tasks cancelled.
        """
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
        """List all tracked tasks.

        Returns:
            list[SubAgentTask]: OUT: All task objects.
        """
        return list(self.tasks.values())

    def list_snapshots(self) -> list[dict[str, Any]]:
        """Return snapshots of all tracked tasks.

        Returns:
            list[dict[str, Any]]: OUT: Snapshot dicts for every task.
        """
        return [t.snapshot() for t in self.tasks.values()]

    def get_by_name(self, name: str) -> SubAgentTask | None:
        """Look up a task by its registered name.

        Args:
            name (str): IN: Registered task name. OUT: Resolved via internal name map.

        Returns:
            SubAgentTask | None: OUT: The matching task, or ``None``.
        """
        task_id = self._by_name.get(name)
        return self.tasks.get(task_id) if task_id else None

    def shutdown(self) -> None:
        """Cancel all active tasks and shut down the thread pool."""
        for task in self.tasks.values():
            if task.status in ("running", "pending"):
                task._cancel_flag = True
        self._pool.shutdown(wait=True)

    def summary(self) -> str:
        """Return a Markdown summary of all tasks.

        Returns:
            str: OUT: Formatted summary with counts and per-task status lines.
        """
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
    """Run the agent streaming loop for a sub-agent task.

    Args:
        prompt (str): IN: User prompt for the sub-agent. OUT: Passed to the loop.
        config (dict[str, Any]): IN: Runtime configuration. OUT: Passed to the loop.
        system_prompt (str): IN: System prompt text. OUT: Passed to the loop.
        depth (int): IN: Delegation depth. OUT: Passed to the loop.
        task (SubAgentTask): IN: The task being executed. OUT: Used for ID and
            name in emitted events.
        tool_executor (Any): IN: Optional tool executor. OUT: Passed to the loop.
        tool_schemas (list[dict[str, Any]] | None): IN: Optional tool schemas. OUT:
            Passed to the loop.

    Returns:
        str: OUT: Concatenated text output from the streaming loop.
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
    """Apply agent-definition tool limits and block recursive delegation tools.

    Sub-agents should execute the task assigned by the parent. They should not
    inherit top-level orchestration tools that let them spawn their own agent
    swarms or re-trigger active skills such as deepscan.
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
