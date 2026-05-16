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
"""Claude-style coding and agent tools for task execution, subagent management, and workflow orchestration.

This module provides comprehensive tools for file editing, code search, subagent spawning,
task management, and workflow control. These tools enable agents to perform complex
multi-step operations and delegate work to specialized subagents.

Example:
    >>> from xerxes.tools.claude_tools import FileEditTool, AgentTool
    >>> FileEditTool.static_call(file_path="main.py", old_string="x=1", new_string="x=2")
    >>> AgentTool.static_call(prompt="Write a test for this function")
"""

from __future__ import annotations

import difflib
import json
import os
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..types import AgentBaseFn


def _unified_diff(old: str, new: str, filename: str = "", context: int = 3) -> str:
    """Generate a unified diff string between two text contents.

    Args:
        old: Original text content.
        new: Modified text content.
        filename: Optional filename for diff headers.
        context: Number of context lines around changes.

    Returns:
        Formatted unified diff string.
    """
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filename}" if filename else "a",
        tofile=f"b/{filename}" if filename else "b",
        n=context,
    )
    result = "".join(diff)
    lines = result.split("\n")
    if len(lines) > 80:
        result = "\n".join(lines[:80]) + f"\n... ({len(lines) - 80} more lines)"
    return result


class FileEditTool(AgentBaseFn):
    """Edit a file by replacing text with new content.

    Provides precise text replacement in files with support for multiple occurrences
    and automatic diff generation for transparency.

    Example:
        >>> FileEditTool.static_call(
        ...     file_path="config.py",
        ...     old_string="DEBUG = True",
        ...     new_string="DEBUG = False"
        ... )
    """

    @staticmethod
    def static_call(
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        **context_variables,
    ) -> str:
        """Replace text in a file with new content.

        Args:
            file_path: Path to the file to edit.
            old_string: Text to find and replace. Must be an exact match.
            new_string: Replacement text.
            replace_all: If True, replace all occurrences. Defaults to False.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Success message with diff summary, or error message.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If old_string is not found or matches new_string.
        """
        p = Path(file_path).expanduser().resolve()
        if not p.exists():
            return f"Error: file not found: {file_path}"

        content = p.read_text(errors="replace")
        count = content.count(old_string)

        if count == 0:
            return "Error: old_string not found in file."
        if count > 1 and not replace_all:
            return (
                f"Error: old_string appears {count} times. "
                "Provide more surrounding context to make it unique, or set replace_all=true."
            )

        if old_string == new_string:
            return "Error: old_string and new_string are identical."

        old_content = content
        new_content = (
            content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
        )

        p.write_text(new_content)
        diff = _unified_diff(old_content, new_content, p.name)
        replacements = count if replace_all else 1
        return f"Applied {replacements} replacement(s) to {p.name}:\n\n{diff}"


class GlobTool(AgentBaseFn):
    """Find files matching a glob pattern.

    Recursively or non-recursively searches for files matching the pattern.

    Example:
        >>> GlobTool.static_call(pattern="**/*.py")
    """

    @staticmethod
    def static_call(
        pattern: str,
        path: str | None = None,
        **context_variables,
    ) -> str:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "*.py", "**/*.txt").
            path: Directory to search in. Defaults to current working directory.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            List of matching file paths, or message if no matches found.
        """
        base = Path(path).expanduser().resolve() if path else Path.cwd()
        try:
            matches = sorted(base.glob(pattern))
            if not matches:
                return "No files matched."
            paths = [str(m) for m in matches[:500]]
            result = "\n".join(paths)
            if len(matches) > 500:
                result += f"\n... ({len(matches) - 500} more matches)"
            return result
        except Exception as e:
            return f"Error: {e}"


class GrepTool(AgentBaseFn):
    """Search file contents using pattern matching.

    Supports both ripgrep (preferred) and standard grep with various output formats.

    Example:
        >>> GrepTool.static_call(pattern="TODO", case_insensitive=True)
    """

    @staticmethod
    def static_call(
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        output_mode: str = "files_with_matches",
        case_insensitive: bool = False,
        context: int = 0,
        **context_variables,
    ) -> str:
        """Search file contents for a pattern.

        Args:
            pattern: Regex or literal string to search for.
            path: Directory or file to search. Defaults to current directory.
            glob: Filter files by glob pattern (e.g., "*.py").
            output_mode: Output format. Options: 'files_with_matches' (default),
                'count', 'content' (line numbers).
            case_insensitive: Make search case-insensitive.
            context: Number of lines of context around matches.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Matching lines or file list, depending on output_mode.
        """
        use_rg = _has_ripgrep()
        cmd: list[str] = ["rg" if use_rg else "grep", "--no-heading"]

        if case_insensitive:
            cmd.append("-i")
        if output_mode == "files_with_matches":
            cmd.append("-l")
        elif output_mode == "count":
            cmd.append("-c")
        else:
            cmd.append("-n")
            if context:
                cmd.extend(["-C", str(context)])

        if glob:
            if use_rg:
                cmd.extend(["--glob", glob])
            else:
                cmd.extend(["--include", glob])

        if use_rg:
            cmd.append("--no-ignore-vcs")
        else:
            cmd.append("-r")

        cmd.append(pattern)
        cmd.append(path or str(Path.cwd()))

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            out = r.stdout.strip()
            if not out:
                return "No matches found."
            return out[:20000]
        except FileNotFoundError:
            return "Error: neither rg nor grep found on PATH."
        except subprocess.TimeoutExpired:
            return "Error: search timed out after 30s."
        except Exception as e:
            return f"Error: {e}"


def _has_ripgrep() -> bool:
    """Check if ripgrep is available on the system.

    Returns:
        True if ripgrep (rg) is installed and accessible.
    """
    try:
        subprocess.run(["rg", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


_agent_manager = None


def _build_subagent_system_prompt(
    base: str = "You are a helpful AI assistant.",
    *,
    include_active_skills: bool = False,
) -> str:
    """Build system prompt for spawned subagents.

    Args:
        base: Base system prompt.
        include_active_skills: Whether to include active skills in prompt.

    Returns:
        Complete system prompt for subagent.
    """
    if not include_active_skills:
        return base

    from ..extensions.skills import get_active_skills
    from ..tools.agent_meta_tools import _skill_registry

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
        from ..agents.subagent_manager import SubAgentManager
        from ..runtime.bridge import build_tool_executor, populate_registry

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
    smart_q = {"“": '"', "”": '"', "‘": "'", "’": "'"}
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
        from ..agents.definitions import get_agent_definition
        from ..runtime.config_context import get_inheritable

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
        from ..agents.definitions import get_agent_definition
        from ..runtime.config_context import get_inheritable

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
        """Get task status and snapshot.

        Args:
            task_id: ID or name of the task to query.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            JSON snapshot of task status and result.
        """
        mgr = _get_agent_manager()
        task = mgr.tasks.get(task_id) or mgr.get_by_name(task_id)
        if not task:
            return f"Error: task '{task_id}' not found."
        return json.dumps(task.snapshot(), indent=2)


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

        from ..runtime.session_context import get_active_session

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


_todo_items: list[dict[str, str]] = []


class TodoWriteTool(AgentBaseFn):
    """Manage a persistent todo list.

    Tracks tasks with status indicators (pending, in_progress, completed).

    Example:
        >>> TodoWriteTool.static_call(
        ...     todos=[
        ...         {"content": "Write tests", "status": "in_progress"},
        ...         {"content": "Deploy", "status": "pending"}
        ...     ]
        ... )
    """

    @staticmethod
    def static_call(
        todos: str | list[dict[str, str]],
        **context_variables,
    ) -> str:
        """Update the todo list.

        Args:
            todos: JSON array of todo items with 'content' and 'status' keys,
                or a JSON string to be parsed.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Formatted todo list with progress summary.
        """
        global _todo_items
        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except json.JSONDecodeError:
                return "Error: todos must be a JSON array of {content, status} objects."

        if not isinstance(todos, list):
            return "Error: todos must be a JSON array of {content, status} objects."
        _todo_items = list(todos)

        lines = ["# Todo List", ""]
        for i, item in enumerate(_todo_items, 1):
            status = item.get("status", "pending")
            content = item.get("content", "")
            icon = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}.get(status, "[ ]")
            lines.append(f"{i}. {icon} {content}")

        done = sum(1 for t in _todo_items if t.get("status") == "completed")
        total = len(_todo_items)
        lines.append(f"\nProgress: {done}/{total}")
        return "\n".join(lines)


_ask_user_question_callback: Callable[[str], str] | None = None


def set_ask_user_question_callback(cb: Callable[[str], str] | None) -> None:
    """Set the callback for user questions.

    Args:
        cb: Callback function that takes a question string and returns an answer.
    """
    global _ask_user_question_callback
    _ask_user_question_callback = cb


class AskUserQuestionTool(AgentBaseFn):
    """Ask a question to the human user; blocks until they answer in the TUI.

    Always interactive — the daemon registers a callback at startup that
    drives the TUI's :class:`QuestionRequestPanel`. If the callback is
    somehow not wired (misconfigured host, broken bootstrap), the tool
    returns a loud error instead of silently echoing the question back
    to the model — the previous "non-interactive fallback" let real bugs
    masquerade as features.

    Example:
        >>> AskUserQuestionTool.static_call(question="Should I proceed with deployment?")
    """

    @staticmethod
    def static_call(
        question: str,
        **context_variables,
    ) -> str:
        """Block on the user's answer and return it as text."""
        global _ask_user_question_callback
        # The daemon installs the callback at startup; in any sane process
        # this is non-None by the time a tool can dispatch. If it's missing
        # we raise instead of fabricating an answer — the LLM would otherwise
        # read the fake string back as if the user had typed it.
        if _ask_user_question_callback is None:
            raise RuntimeError(
                "AskUserQuestion callback was never registered; daemon bootstrap is broken"
            )
        return _ask_user_question_callback(question)


class EnterPlanModeTool(AgentBaseFn):
    """Enter plan mode where actions are described but not executed.

    Use for reviewing and validating task plans before execution.

    Example:
        >>> EnterPlanModeTool.static_call()
    """

    @staticmethod
    def static_call(**context_variables) -> str:
        """Enter plan mode.

        Args:
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Confirmation message.
        """
        return "Entered plan mode. Describe your plan without executing actions."


class ExitPlanModeTool(AgentBaseFn):
    """Exit plan mode and resume normal action execution.

    Example:
        >>> ExitPlanModeTool.static_call()
    """

    @staticmethod
    def static_call(**context_variables) -> str:
        """Exit plan mode.

        Args:
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Confirmation message.
        """
        return "Exited plan mode. Resuming normal execution."


class SetInteractionModeTool(AgentBaseFn):
    """Switch the active interaction mode for future turns.

    Use this when the task should move between code, research, and plan modes.
    """

    @staticmethod
    def static_call(
        mode: str,
        reason: str = "",
        **context_variables,
    ) -> str:
        """Switch interaction mode.

        Args:
            mode: Target mode. One of ``code``, ``researcher``/``research``, or
                ``plan``/``planner``.
            reason: Short reason for the switch.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Confirmation message with the normalized target mode.
        """
        from ..runtime.config_context import emit_event, get_config, set_config

        aliases = {
            "": "code",
            "coding": "code",
            "coder": "code",
            "code": "code",
            "research": "researcher",
            "researcher": "researcher",
            "plan": "plan",
            "planner": "plan",
        }
        normalized = aliases.get((mode or "").strip().lower())
        if normalized is None:
            return "Error: mode must be one of code, researcher, or plan."

        config = get_config()
        config["mode"] = normalized
        config["plan_mode"] = normalized == "plan"
        set_config(config)

        emit_event(
            "interaction_mode_changed",
            {
                "mode": normalized,
                "plan_mode": normalized == "plan",
                "reason": reason,
                "source": "model",
            },
        )
        note = f" Reason: {reason}" if reason else ""
        return f"Interaction mode switched to {normalized}.{note}"


class EnterWorktreeTool(AgentBaseFn):
    """Create a git worktree for isolated task execution.

    Provides a separate working directory with its own branch.

    Example:
        >>> EnterWorktreeTool.static_call(branch_name="feature-task")
    """

    @staticmethod
    def static_call(
        branch_name: str = "",
        **context_variables,
    ) -> str:
        """Create a git worktree.

        Args:
            branch_name: Name for the new branch. Auto-generated if empty.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Success message with worktree path and branch name, or error.
        """
        cwd = os.getcwd()
        try:
            git_root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return "Error: not in a git repository."

        branch = branch_name or f"xerxes-worktree-{uuid.uuid4().hex[:8]}"
        wt_path = tempfile.mkdtemp(prefix="xerxes-wt-")
        os.rmdir(wt_path)

        try:
            subprocess.run(
                ["git", "worktree", "add", "-b", branch, wt_path],
                cwd=git_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return f"Worktree created:\n  Path: {wt_path}\n  Branch: {branch}\n  Base: {git_root}"
        except subprocess.CalledProcessError as e:
            return f"Error creating worktree: {e.stderr}"


class ExitWorktreeTool(AgentBaseFn):
    """Remove a git worktree after task completion.

    Example:
        >>> ExitWorktreeTool.static_call(worktree_path="/tmp/xerxes-wt-abc123")
    """

    @staticmethod
    def static_call(
        worktree_path: str,
        force: bool = False,
        **context_variables,
    ) -> str:
        """Remove a git worktree.

        Args:
            worktree_path: Path to the worktree to remove.
            force: Force removal even with uncommitted changes.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Success message or error.
        """
        cmd = ["git", "worktree", "remove"]
        if force:
            cmd.append("--force")
        cmd.append(worktree_path)

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return f"Worktree removed: {worktree_path}"
        except subprocess.CalledProcessError as e:
            return f"Error: {e.stderr.strip()}"


class ToolSearchTool(AgentBaseFn):
    """Search available tools by name or description.

    Example:
        >>> ToolSearchTool.static_call(query="file edit")
    """

    @staticmethod
    def static_call(
        query: str,
        **context_variables,
    ) -> str:
        """Search for matching tools.

        Args:
            query: Search query for tool names and descriptions.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            List of matching tools with scores.
        """
        from ..runtime.bridge import populate_registry

        registry = populate_registry()
        matches = registry.route(query, limit=10)
        if not matches:
            return "No matching tools found."
        lines = [f"Found {len(matches)} matching tools:", ""]
        for m in matches:
            lines.append(f"- **{m.name}** (score={m.score}) — {m.description[:80]}")
        return "\n".join(lines)


class SkillTool(AgentBaseFn):
    """Invoke a named skill with optional arguments.

    Skills are reusable prompt templates stored in the skills directory.

    Example:
        >>> SkillTool.static_call(skill_name="code-review", args="Check the auth module")
    """

    @staticmethod
    def static_call(
        skill_name: str,
        args: str = "",
        **context_variables,
    ) -> str:
        """Invoke a skill by name.

        Args:
            skill_name: Name of the skill to invoke.
            args: Optional arguments to append to skill prompt.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Formatted skill instructions or error.
        """
        try:
            from ..core.paths import xerxes_subdir
            from ..extensions.skills import SkillRegistry

            registry = SkillRegistry()
            skills_dir = xerxes_subdir("skills")
            project_skills = Path.cwd() / "skills"

            import xerxes as _xerxes_pkg

            _bundled = Path(_xerxes_pkg.__file__).parent / "skills"
            discover_dirs = [str(skills_dir), str(project_skills)]
            if _bundled.is_dir():
                discover_dirs.insert(0, str(_bundled))
            registry.discover(*discover_dirs)

            skill = registry.get(skill_name)
            if skill is None:
                available = [s.name for s in registry.get_all()]
                if available:
                    return f"Skill '{skill_name}' not found. Available: {', '.join(available[:20])}"
                return f"Skill '{skill_name}' not found. No skills discovered."
            prompt = skill.to_prompt_section()
            if args:
                prompt += f"\n\nUser request: {args}"
            return f"[Skill: {skill_name}]\n{prompt}"
        except Exception as e:
            return f"Error invoking skill '{skill_name}': {e}"


class NotebookEditTool(AgentBaseFn):
    """Edit a Jupyter notebook cell.

    Example:
        >>> NotebookEditTool.static_call(
        ...     notebook_path="analysis.ipynb",
        ...     cell_index=2,
        ...     new_source="print('Updated!')"
        ... )
    """

    @staticmethod
    def static_call(
        notebook_path: str,
        cell_index: int,
        new_source: str,
        cell_type: str = "code",
        **context_variables,
    ) -> str:
        """Edit a notebook cell.

        Args:
            notebook_path: Path to the .ipynb file.
            cell_index: Zero-based index of the cell to edit.
            new_source: New content for the cell.
            cell_type: Cell type ('code' or 'markdown').
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Success message or error.
        """
        p = Path(notebook_path).expanduser().resolve()
        if not p.exists():
            return f"Error: notebook not found: {notebook_path}"

        try:
            nb = json.loads(p.read_text())
            cells = nb.get("cells", [])
            if cell_index < 0 or cell_index >= len(cells):
                return f"Error: cell_index {cell_index} out of range (0-{len(cells) - 1})."

            cells[cell_index]["source"] = new_source.splitlines(keepends=True)
            cells[cell_index]["cell_type"] = cell_type
            p.write_text(json.dumps(nb, indent=1) + "\n")
            return f"Updated cell {cell_index} in {p.name} ({cell_type}, {len(new_source)} chars)."
        except json.JSONDecodeError:
            return "Error: invalid notebook format."
        except Exception as e:
            return f"Error: {e}"


class LSPTool(AgentBaseFn):
    """Interface for Language Server Protocol operations.

    Provides code navigation, diagnostics, and refactoring via LSP.

    Example:
        >>> LSPTool.static_call(action="definition", file_path="main.py", line=10)
    """

    @staticmethod
    def static_call(
        action: str,
        file_path: str = "",
        line: int = 0,
        character: int = 0,
        **context_variables,
    ) -> str:
        """Execute an LSP action.

        Args:
            action: LSP action (e.g., 'definition', 'references', 'hover').
            file_path: File to operate on.
            line: Line number (1-indexed).
            character: Character position.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Informational message (LSP requires active server in TUI).
        """
        return (
            f"[LSP:{action}] file={file_path} line={line} char={character}\n"
            "LSP tool requires an active language server. In the TUI, this is "
            "handled by the IDE integration layer. Use Grep/Glob for code search instead."
        )


class MCPTool(AgentBaseFn):
    """Interface for Model Context Protocol tools.

    Allows invoking tools from configured MCP servers.

    Example:
        >>> MCPTool.static_call(server_name="filesystem", tool_name="read_file")
    """

    @staticmethod
    def static_call(
        server_name: str,
        tool_name: str,
        arguments: str | dict | None = None,
        **context_variables,
    ) -> str:
        """Call an MCP tool.

        Args:
            server_name: Name of the MCP server.
            tool_name: Name of the tool to call.
            arguments: Tool arguments as dict or JSON string.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Tool result or informational message.
        """
        import importlib.util

        if importlib.util.find_spec("xerxes.mcp") is not None:
            return (
                f"[MCP] server={server_name} tool={tool_name}\n"
                "Use xerxes.mcp.MCPManager for async MCP tool invocation. "
                "This tool is a placeholder for the synchronous tool interface."
            )
        return "Error: xerxes.mcp module not available. Install xerxes[mcp]."


class ListMcpResourcesTool(AgentBaseFn):
    """List available resources from MCP servers.

    Example:
        >>> ListMcpResourcesTool.static_call(server_name="database")
    """

    @staticmethod
    def static_call(server_name: str = "", **context_variables) -> str:
        """List MCP resources.

        Args:
            server_name: Filter by server name, or empty for all servers.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            List of available resources.
        """
        return (
            f"[MCP Resources] server={server_name or '(all)'}\n"
            "Use xerxes.mcp.MCPManager.list_resources() for async MCP resource listing."
        )


class ReadMcpResourceTool(AgentBaseFn):
    """Read a specific resource from an MCP server.

    Example:
        >>> ReadMcpResourceTool.static_call(server_name="config", uri="file:///settings.json")
    """

    @staticmethod
    def static_call(
        server_name: str,
        uri: str,
        **context_variables,
    ) -> str:
        """Read an MCP resource.

        Args:
            server_name: MCP server name.
            uri: Resource URI to read.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Resource content or error.
        """
        return (
            f"[MCP Read] server={server_name} uri={uri}\n"
            "Use xerxes.mcp.MCPManager.read_resource() for async MCP resource reading."
        )


class RemoteTriggerTool(AgentBaseFn):
    """Trigger a named remote endpoint.

    Example:
        >>> RemoteTriggerTool.static_call(trigger_name="notify-slack", payload="Build complete")
    """

    @staticmethod
    def static_call(
        trigger_name: str,
        payload: str = "",
        **context_variables,
    ) -> str:
        """Trigger a remote endpoint.

        Args:
            trigger_name: Name of the trigger to invoke.
            payload: Data to send with the trigger.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Status message.
        """
        return f"[RemoteTrigger] name={trigger_name} payload={payload[:100]}\nRemote triggers require configured remote endpoints."


class ScheduleCronTool(AgentBaseFn):
    """Schedule a task for cron-based execution.

    Example:
        >>> ScheduleCronTool.static_call(schedule="0 9 * * *", prompt="Daily report")
    """

    @staticmethod
    def static_call(
        schedule: str,
        prompt: str,
        name: str = "",
        **context_variables,
    ) -> str:
        """Schedule a cron task.

        Args:
            schedule: Cron expression (e.g., "0 9 * * *").
            prompt: Task prompt for the scheduled execution.
            name: Optional name for the scheduled task.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Status message.
        """
        return (
            f"[ScheduleCron] schedule={schedule} name={name or '(unnamed)'}\n"
            f"Prompt: {prompt[:100]}\n"
            "Cron scheduling requires a persistent scheduler service."
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
        from ..agents.definitions import get_agent_definition
        from ..runtime.config_context import emit_event, get_inheritable

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


class PlanTool(AgentBaseFn):
    """Create and optionally execute a multi-step plan.

    Breaks down complex tasks into parallelizable steps with agent assignments.

    Example:
        >>> PlanTool.static_call(
        ...     objective="Build and deploy web app",
        ...     execute=True
        ... )
    """

    @staticmethod
    def static_call(
        objective: str,
        execute: bool = True,
        **context_variables,
    ) -> str:
        """Create and optionally execute a task plan.

        Args:
            objective: High-level task description.
            execute: Whether to execute the plan. Defaults to True.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Formatted plan and execution results.
        """
        from ..agents.definitions import get_agent_definition, list_agent_definitions
        from ..runtime.config_context import emit_event, get_inheritable

        config = get_inheritable()
        mgr = _get_agent_manager()

        agent_defs = list_agent_definitions()
        agents_desc = "\n".join(f"- **{d.name}**: {d.description}" for d in agent_defs)

        plan_prompt = f"""You are a task planner. Break down this objective into discrete steps.

{agents_desc}

{objective}

<plan>
  <step id="1" agent="general-purpose" depends="">
    <description>First step description</description>
  </step>
  <step id="2" agent="coder" depends="">
    <description>Second step (independent, can run in parallel with step 1)</description>
  </step>
  <step id="3" agent="reviewer" depends="1,2">
    <description>Third step (depends on steps 1 and 2 completing first)</description>
  </step>
</plan>

Rules:
- Use the most appropriate agent type for each step.
- Mark dependencies accurately — steps with no depends="" run in parallel.
- Keep steps focused and actionable.
- Output ONLY the <plan> XML, nothing else.
"""

        from ..streaming.events import AgentState, TextChunk
        from ..streaming.loop import run

        state = AgentState()
        plan_text_parts: list[str] = []
        for event in run(
            user_message=plan_prompt,
            state=state,
            config=config,
            system_prompt="You are a precise task planner.",
        ):
            if isinstance(event, TextChunk):
                plan_text_parts.append(event.text)

        plan_xml = "".join(plan_text_parts)

        steps = _parse_plan_xml(plan_xml)
        if not steps:
            return f"Failed to parse plan. Raw output:\n{plan_xml}"

        plan_summary = "# Execution Plan\n\n"
        for s in steps:
            deps = f" (depends on: {s['depends']})" if s["depends"] else ""
            plan_summary += f"**Step {s['id']}** [{s['agent']}]{deps}: {s['description']}\n"

        emit_event(
            "plan_created",
            {
                "objective": objective,
                "steps": steps,
            },
        )

        if not execute:
            return plan_summary

        completed: dict[str, str] = {}
        results: list[str] = [plan_summary, "\n# Execution Results\n"]

        remaining = list(steps)
        wait_timeout = _subagent_wait_timeout()
        while remaining:
            ready = [s for s in remaining if all(d.strip() in completed for d in s["depends"].split(",") if d.strip())]
            if not ready:
                results.append("\n**Deadlock:** remaining steps have unresolvable dependencies.")
                break

            tasks = []
            for step in ready:
                agent_def = get_agent_definition(step["agent"])

                step_prompt = f"""## Task (Step {step["id"]} of plan)

{step["description"]}

"""
                if step["depends"]:
                    step_prompt += "## Results from previous steps:\n"
                    for dep_id in step["depends"].split(","):
                        dep_id = dep_id.strip()
                        if dep_id in completed:
                            step_prompt += f"\n### Step {dep_id}:\n{completed[dep_id][:1000]}\n"

                emit_event(
                    "plan_step_start",
                    {
                        "step_id": step["id"],
                        "agent": step["agent"],
                        "description": step["description"],
                    },
                )

                task = mgr.spawn(
                    prompt=step_prompt,
                    config=config,
                    system_prompt="You are a helpful AI assistant.",
                    agent_def=agent_def,
                    name=f"plan-step-{step['id']}",
                )
                tasks.append((step, task))

            deadline = None if wait_timeout is None else time.monotonic() + wait_timeout
            still_running = False
            for step, task in tasks:
                remaining_wait = None if deadline is None else max(0.0, deadline - time.monotonic())
                mgr.wait(task.id, timeout=remaining_wait)
                result = task.result or f"(failed: {task.error})"
                results.append(f"\n## Step {step['id']} [{step['agent']}]: {step['description']}")
                results.append(f"Status: {task.status}")
                if task.status in ("pending", "running"):
                    still_running = True
                    results.append("Still running; use TaskGetTool or TaskListTool to check it later.")
                else:
                    completed[step["id"]] = result
                    results.append(result[:2000])

                emit_event(
                    "plan_step_done",
                    {
                        "step_id": step["id"],
                        "status": task.status,
                    },
                )

            remaining = [s for s in remaining if s["id"] not in completed]
            if still_running:
                results.append("\nPlan execution is still running in background; dependent steps were not started yet.")
                break

        return "\n".join(results)


def _parse_plan_xml(xml_text: str) -> list[dict[str, str]]:
    """Parse the <plan> XML format into step dictionaries.

    Args:
        xml_text: XML string containing <plan> element.

    Returns:
        List of step dictionaries with id, agent, depends, description.
    """
    import re

    steps = []
    step_pattern = re.compile(
        r'<step\s+id="([^"]+)"\s+agent="([^"]+)"\s+depends="([^"]*)"[^>]*>'
        r"\s*<description>(.*?)</description>\s*</step>",
        re.DOTALL,
    )
    for match in step_pattern.finditer(xml_text):
        steps.append(
            {
                "id": match.group(1),
                "agent": match.group(2),
                "depends": match.group(3),
                "description": match.group(4).strip(),
            }
        )
    return steps


__all__ = [
    "AgentTool",
    "AskUserQuestionTool",
    "EnterPlanModeTool",
    "EnterWorktreeTool",
    "ExitPlanModeTool",
    "ExitWorktreeTool",
    "FileEditTool",
    "GlobTool",
    "GrepTool",
    "HandoffTool",
    "LSPTool",
    "ListMcpResourcesTool",
    "MCPTool",
    "NotebookEditTool",
    "PlanTool",
    "ReadMcpResourceTool",
    "RemoteTriggerTool",
    "ScheduleCronTool",
    "SendMessageTool",
    "SetInteractionModeTool",
    "SkillTool",
    "TaskCreateTool",
    "TaskGetTool",
    "TaskListTool",
    "TaskOutputTool",
    "TaskStopTool",
    "TaskUpdateTool",
    "TodoWriteTool",
    "ToolSearchTool",
]
