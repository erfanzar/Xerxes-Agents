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
"""Claude tools module for Xerxes.

Exports:
    - FileEditTool
    - GlobTool
    - GrepTool
    - AgentTool
    - SendMessageTool
    - TaskCreateTool
    - SpawnAgents
    - TaskGetTool
    - TaskListTool
    - TaskOutputTool
    - ... and 20 more."""

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
    """Internal helper to unified diff.

    Args:
        old (str): IN: old. OUT: Consumed during execution.
        new (str): IN: new. OUT: Consumed during execution.
        filename (str, optional): IN: filename. Defaults to ''. OUT: Consumed during execution.
        context (int, optional): IN: context. Defaults to 3. OUT: Consumed during execution.
    Returns:
        str: OUT: Result of the operation."""

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
    """File edit tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            file_path (str): IN: file path. OUT: Consumed during execution.
            old_string (str): IN: old string. OUT: Consumed during execution.
            new_string (str): IN: new string. OUT: Consumed during execution.
            replace_all (bool, optional): IN: replace all. Defaults to False. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

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
    """Glob tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        pattern: str,
        path: str | None = None,
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            pattern (str): IN: pattern. OUT: Consumed during execution.
            path (str | None, optional): IN: path. Defaults to None. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

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
    """Grep tool.

    Inherits from: AgentBaseFn
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
        """Static call.

        Args:
            pattern (str): IN: pattern. OUT: Consumed during execution.
            path (str | None, optional): IN: path. Defaults to None. OUT: Consumed during execution.
            glob (str | None, optional): IN: glob. Defaults to None. OUT: Consumed during execution.
            output_mode (str, optional): IN: output mode. Defaults to 'files_with_matches'. OUT: Consumed during execution.
            case_insensitive (bool, optional): IN: case insensitive. Defaults to False. OUT: Consumed during execution.
            context (int, optional): IN: context. Defaults to 0. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

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
    """Internal helper to has ripgrep.

    Returns:
        bool: OUT: Result of the operation."""

    try:
        subprocess.run(["rg", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


_agent_manager = None


def _build_subagent_system_prompt(base: str = "You are a helpful AI assistant.") -> str:
    """Internal helper to build subagent system prompt.

    Args:
        base (str, optional): IN: base. Defaults to 'You are a helpful AI assistant.'. OUT: Consumed during execution.
    Returns:
        str: OUT: Result of the operation."""

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
    """Internal helper to get agent manager.

    Returns:
        Any: OUT: Result of the operation."""

    global _agent_manager
    if _agent_manager is None:
        from ..agents.subagent_manager import SubAgentManager
        from ..runtime.bridge import build_tool_executor, populate_registry

        _agent_manager = SubAgentManager()
        registry = populate_registry()
        _agent_manager._tool_executor = build_tool_executor(registry=registry)
        _agent_manager._tool_schemas = registry.tool_schemas()
    return _agent_manager


def _spawn_agents_wait_timeout(value: float | int | str | None = None) -> float | None:
    """Return the bounded wait for SpawnAgents joins."""
    raw = value if value is not None else os.environ.get("XERXES_SPAWN_AGENTS_WAIT_TIMEOUT", "120")
    if raw is None:
        return 120.0
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return 120.0
    if timeout <= 0:
        return None
    return timeout


class AgentTool(AgentBaseFn):
    """Agent tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        prompt: str,
        subagent_type: str = "general-purpose",
        isolation: str = "",
        name: str = "",
        model: str = "",
        wait: bool = True,
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            prompt (str): IN: prompt. OUT: Consumed during execution.
            subagent_type (str, optional): IN: subagent type. Defaults to 'general-purpose'. OUT: Consumed during execution.
            isolation (str, optional): IN: isolation. Defaults to ''. OUT: Consumed during execution.
            name (str, optional): IN: name. Defaults to ''. OUT: Consumed during execution.
            model (str, optional): IN: model. Defaults to ''. OUT: Consumed during execution.
            wait (bool, optional): IN: wait. Defaults to True. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

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

        if wait and task.status not in ("failed",):
            mgr.wait(task.id, timeout=None)

        if task.status == "completed" and task.result is not None:
            return task.result
        if task.status == "failed":
            return f"Agent failed: {task.error}"
        if task.status == "cancelled":
            return "[Sub-agent was cancelled.]"

        return json.dumps(task.snapshot(), indent=2)


class SendMessageTool(AgentBaseFn):
    """Send message tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        target: str,
        message: str,
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            target (str): IN: target. OUT: Consumed during execution.
            message (str): IN: message. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

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
    """Task create tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        prompt: str,
        name: str = "",
        subagent_type: str = "general-purpose",
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            prompt (str): IN: prompt. OUT: Consumed during execution.
            name (str, optional): IN: name. Defaults to ''. OUT: Consumed during execution.
            subagent_type (str, optional): IN: subagent type. Defaults to 'general-purpose'. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

        return AgentTool.static_call(
            prompt=prompt,
            subagent_type=subagent_type,
            name=name,
            wait=False,
        )


class SpawnAgents(AgentBaseFn):
    """Spawn agents.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        agents: list[dict[str, str]] | str,
        wait: bool = True,
        timeout: float | None = None,
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            agents (list[dict[str, str]] | str): IN: agents. OUT: Consumed during execution.
            wait (bool, optional): IN: wait. Defaults to True. OUT: Consumed during execution.
            timeout (float | None, optional): IN: max seconds to wait for all agents. Defaults to env or 120.
                OUT: Prevents the parent tool call from occupying the TUI indefinitely.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

        from ..agents.definitions import get_agent_definition
        from ..runtime.config_context import get_inheritable

        if isinstance(agents, str):
            try:
                agents = json.loads(agents)
            except Exception:
                return f"[Error: agents must be a JSON array of objects, got: {agents[:200]}]"
        if not isinstance(agents, list):
            return f"[Error: agents must be a list, got {type(agents).__name__}]"

        mgr = _get_agent_manager()
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
    """Task get tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(task_id: str, **context_variables) -> str:
        """Static call.

        Args:
            task_id (str): IN: task id. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

        mgr = _get_agent_manager()
        task = mgr.tasks.get(task_id) or mgr.get_by_name(task_id)
        if not task:
            return f"Error: task '{task_id}' not found."
        return json.dumps(task.snapshot(), indent=2)


class TaskListTool(AgentBaseFn):
    """Task list tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(**context_variables) -> str:
        """Static call.

        Args:
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

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
    """Task output tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(task_id: str, **context_variables) -> str:
        """Static call.

        Args:
            task_id (str): IN: task id. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

        mgr = _get_agent_manager()
        result = mgr.get_result(task_id)
        if result is None:
            task = mgr.get_by_name(task_id)
            if task:
                result = task.result
        return result or f"No output for task '{task_id}' (may still be running)."


class TaskStopTool(AgentBaseFn):
    """Task stop tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(task_id: str, **context_variables) -> str:
        """Static call.

        Args:
            task_id (str): IN: task id. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

        mgr = _get_agent_manager()
        ok = mgr.cancel(task_id)
        return f"Task '{task_id}' cancelled." if ok else f"Could not cancel task '{task_id}'."


class TaskUpdateTool(AgentBaseFn):
    """Task update tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        task_id: str,
        message: str,
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            task_id (str): IN: task id. OUT: Consumed during execution.
            message (str): IN: message. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

        return SendMessageTool.static_call(target=task_id, message=message)


_todo_items: list[dict[str, str]] = []


class TodoWriteTool(AgentBaseFn):
    """Todo write tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        todos: str | list[dict[str, str]],
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            todos (str | list[dict[str, str]]): IN: todos. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

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
    """Set the ask user question callback.

    Args:
        cb (Callable[[str], str] | None): IN: cb. OUT: Consumed during execution."""

    global _ask_user_question_callback
    _ask_user_question_callback = cb


class AskUserQuestionTool(AgentBaseFn):
    """Ask user question tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        question: str,
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            question (str): IN: question. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

        global _ask_user_question_callback
        if _ask_user_question_callback is not None:
            return _ask_user_question_callback(question)
        return f"[AskUserQuestion] {question}\n(Waiting for user response — in non-interactive mode, this returns the question itself.)"


class EnterPlanModeTool(AgentBaseFn):
    """Enter plan mode tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(**context_variables) -> str:
        """Static call.

        Args:
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

        return "Entered plan mode. Describe your plan without executing actions."


class ExitPlanModeTool(AgentBaseFn):
    """Exit plan mode tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(**context_variables) -> str:
        """Static call.

        Args:
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

        return "Exited plan mode. Resuming normal execution."


class EnterWorktreeTool(AgentBaseFn):
    """Enter worktree tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        branch_name: str = "",
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            branch_name (str, optional): IN: branch name. Defaults to ''. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

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
    """Exit worktree tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        worktree_path: str,
        force: bool = False,
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            worktree_path (str): IN: worktree path. OUT: Consumed during execution.
            force (bool, optional): IN: force. Defaults to False. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

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
    """Tool search tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        query: str,
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            query (str): IN: query. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

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
    """Skill tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        skill_name: str,
        args: str = "",
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            skill_name (str): IN: skill name. OUT: Consumed during execution.
            args (str, optional): IN: args. Defaults to ''. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

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
    """Notebook edit tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        notebook_path: str,
        cell_index: int,
        new_source: str,
        cell_type: str = "code",
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            notebook_path (str): IN: notebook path. OUT: Consumed during execution.
            cell_index (int): IN: cell index. OUT: Consumed during execution.
            new_source (str): IN: new source. OUT: Consumed during execution.
            cell_type (str, optional): IN: cell type. Defaults to 'code'. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

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
    """Lsptool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        action: str,
        file_path: str = "",
        line: int = 0,
        character: int = 0,
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            action (str): IN: action. OUT: Consumed during execution.
            file_path (str, optional): IN: file path. Defaults to ''. OUT: Consumed during execution.
            line (int, optional): IN: line. Defaults to 0. OUT: Consumed during execution.
            character (int, optional): IN: character. Defaults to 0. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

        return (
            f"[LSP:{action}] file={file_path} line={line} char={character}\n"
            "LSP tool requires an active language server. In the TUI, this is "
            "handled by the IDE integration layer. Use Grep/Glob for code search instead."
        )


class MCPTool(AgentBaseFn):
    """Mcptool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        server_name: str,
        tool_name: str,
        arguments: str | dict | None = None,
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            server_name (str): IN: server name. OUT: Consumed during execution.
            tool_name (str): IN: tool name. OUT: Consumed during execution.
            arguments (str | dict | None, optional): IN: arguments. Defaults to None. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

        import importlib.util

        if importlib.util.find_spec("xerxes.mcp") is not None:
            return (
                f"[MCP] server={server_name} tool={tool_name}\n"
                "Use xerxes.mcp.MCPManager for async MCP tool invocation. "
                "This tool is a placeholder for the synchronous tool interface."
            )
        return "Error: xerxes.mcp module not available. Install xerxes[mcp]."


class ListMcpResourcesTool(AgentBaseFn):
    """List mcp resources tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(server_name: str = "", **context_variables) -> str:
        """Static call.

        Args:
            server_name (str, optional): IN: server name. Defaults to ''. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

        return (
            f"[MCP Resources] server={server_name or '(all)'}\n"
            "Use xerxes.mcp.MCPManager.list_resources() for async MCP resource listing."
        )


class ReadMcpResourceTool(AgentBaseFn):
    """Read mcp resource tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        server_name: str,
        uri: str,
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            server_name (str): IN: server name. OUT: Consumed during execution.
            uri (str): IN: uri. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

        return (
            f"[MCP Read] server={server_name} uri={uri}\n"
            "Use xerxes.mcp.MCPManager.read_resource() for async MCP resource reading."
        )


class RemoteTriggerTool(AgentBaseFn):
    """Remote trigger tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        trigger_name: str,
        payload: str = "",
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            trigger_name (str): IN: trigger name. OUT: Consumed during execution.
            payload (str, optional): IN: payload. Defaults to ''. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

        return f"[RemoteTrigger] name={trigger_name} payload={payload[:100]}\nRemote triggers require configured remote endpoints."


class ScheduleCronTool(AgentBaseFn):
    """Schedule cron tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        schedule: str,
        prompt: str,
        name: str = "",
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            schedule (str): IN: schedule. OUT: Consumed during execution.
            prompt (str): IN: prompt. OUT: Consumed during execution.
            name (str, optional): IN: name. Defaults to ''. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

        return (
            f"[ScheduleCron] schedule={schedule} name={name or '(unnamed)'}\n"
            f"Prompt: {prompt[:100]}\n"
            "Cron scheduling requires a persistent scheduler service."
        )


class HandoffTool(AgentBaseFn):
    """Handoff tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        target_agent: str,
        reason: str,
        context_summary: str = "",
        prompt: str = "",
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            target_agent (str): IN: target agent. OUT: Consumed during execution.
            reason (str): IN: reason. OUT: Consumed during execution.
            context_summary (str, optional): IN: context summary. Defaults to ''. OUT: Consumed during execution.
            prompt (str, optional): IN: prompt. Defaults to ''. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

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

        mgr.wait(task.id, timeout=None)

        if task.status == "completed" and task.result:
            return task.result
        if task.status == "failed":
            return f"Handoff to {target_agent} failed: {task.error}"
        return json.dumps(task.snapshot(), indent=2)


class PlanTool(AgentBaseFn):
    """Plan tool.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        objective: str,
        execute: bool = True,
        **context_variables,
    ) -> str:
        """Static call.

        Args:
            objective (str): IN: objective. OUT: Consumed during execution.
            execute (bool, optional): IN: execute. Defaults to True. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

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

            for step, task in tasks:
                mgr.wait(task.id, timeout=None)
                result = task.result or f"(failed: {task.error})"
                completed[step["id"]] = result
                results.append(f"\n## Step {step['id']} [{step['agent']}]: {step['description']}")
                results.append(f"Status: {task.status}")
                results.append(result[:2000])

                emit_event(
                    "plan_step_done",
                    {
                        "step_id": step["id"],
                        "status": task.status,
                    },
                )

            remaining = [s for s in remaining if s["id"] not in completed]

        return "\n".join(results)


def _parse_plan_xml(xml_text: str) -> list[dict[str, str]]:
    """Internal helper to parse plan xml.

    Args:
        xml_text (str): IN: xml text. OUT: Consumed during execution.
    Returns:
        list[dict[str, str]]: OUT: Result of the operation."""

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
