# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""Claude Code-parity tool implementations.

This module implements the full Claude Code agent tool surface that was
cataloged in the claw-code reference snapshots. Each tool class extends
``AgentBaseFn`` and provides a ``static_call`` method matching the
Claude Code tool interface.

Tools implemented:

**File operations:**
- FileEditTool — Exact string replacement with uniqueness check and diff output
- GlobTool — File pattern matching via Python glob
- GrepTool — Regex search with ripgrep fallback

**Agent management:**
- AgentTool — Spawn typed sub-agents with worktree isolation
- SendMessageTool — Queue messages to running sub-agents
- TaskCreateTool / TaskGetTool / TaskListTool / TaskOutputTool / TaskStopTool / TaskUpdateTool

**Workflow:**
- TodoWriteTool — Structured task list management
- AskUserQuestionTool — Prompt user for input
- EnterPlanModeTool / ExitPlanModeTool — Plan mode toggling
- EnterWorktreeTool / ExitWorktreeTool — Git worktree isolation
- ToolSearchTool — Search available tools by keyword
- SkillTool — Invoke named skills

**Advanced:**
- NotebookEditTool — Jupyter notebook cell editing
- LSPTool — Language Server Protocol queries
- MCPTool / ListMcpResourcesTool / ReadMcpResourceTool — MCP integration
- RemoteTriggerTool / ScheduleCronTool — Remote/scheduled execution
"""

from __future__ import annotations

import difflib
import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from ..types import AgentBaseFn


def _unified_diff(old: str, new: str, filename: str = "", context: int = 3) -> str:
    """Generate a unified diff between old and new content."""
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
    """Exact string replacement in files (Claude Code Edit tool).

    Replaces an exact substring in a file. The old_string must be unique
    in the file unless replace_all is True. Shows a unified diff of changes.
    """

    @staticmethod
    def static_call(
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        **context_variables,
    ) -> str:
        """
        Replace exact text in a file.

        Performs an exact string replacement in the specified file. The old_string
        must appear in the file. If it appears more than once and replace_all is
        False, the operation fails with an error asking for more context.

        Args:
            file_path: Path to the file to edit.
            old_string: The exact text to find and replace.
            new_string: The replacement text. Must differ from old_string.
            replace_all: If True, replace all occurrences. If False (default),
                old_string must be unique in the file.

        Returns:
            A unified diff showing the changes made.
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
    """File pattern matching via glob (Claude Code Glob tool).

    Finds files matching a glob pattern. Returns sorted paths.
    """

    @staticmethod
    def static_call(
        pattern: str,
        path: str | None = None,
        **context_variables,
    ) -> str:
        """
        Find files matching a glob pattern.

        Use this tool to discover files in the workspace by pattern. Supports
        standard glob syntax like ``**/*.py``, ``src/**/*.ts``, ``*.json``.

        Args:
            pattern: Glob pattern to match (e.g. ``**/*.py``).
            path: Base directory to search from. Defaults to current directory.

        Returns:
            Newline-separated list of matching file paths, sorted by path.
            Returns at most 500 matches.
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
    """Regex search with ripgrep fallback (Claude Code Grep tool).

    Searches file contents using regex patterns. Uses ripgrep (rg) if
    available, falls back to grep.
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
        """
        Search file contents with regex.

        Searches for a regex pattern across files. Uses ripgrep (rg) for speed
        when available, falls back to grep.

        Args:
            pattern: Regex pattern to search for.
            path: Directory or file to search in. Defaults to current directory.
            glob: File glob filter (e.g. ``*.py``, ``*.{ts,tsx}``).
            output_mode: One of ``files_with_matches`` (default, file paths only),
                ``content`` (matching lines), or ``count`` (match counts).
            case_insensitive: Case-insensitive search.
            context: Lines of context around matches (for content mode).

        Returns:
            Search results as text. Truncated to 20000 chars.
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
    """Check if ripgrep (rg) is on PATH."""
    try:
        subprocess.run(["rg", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


_agent_manager = None


def _get_agent_manager():
    """Get or create the global SubAgentManager."""
    global _agent_manager
    if _agent_manager is None:
        from ..agents.subagent_manager import SubAgentManager
        from ..runtime.bridge import build_tool_executor, populate_registry

        _agent_manager = SubAgentManager()
        registry = populate_registry()
        _agent_manager._tool_executor = build_tool_executor(registry=registry)
        _agent_manager._tool_schemas = registry.tool_schemas()
    return _agent_manager


class AgentTool(AgentBaseFn):
    """Spawn sub-agents with typed definitions and worktree isolation."""

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
        """
        Spawn a sub-agent to handle a task autonomously.

        Launches a new agent in a thread to process the given prompt. The agent
        can use a specialized type (coder, reviewer, researcher, tester, planner)
        and optionally run in an isolated git worktree.

        Args:
            prompt: The task description for the sub-agent.
            subagent_type: Agent type name (e.g. ``coder``, ``reviewer``).
            isolation: ``""`` for normal, ``"worktree"`` for git isolation.
            name: Human-readable name for addressing via SendMessage.
            model: Model override. Empty = inherit from parent.
            wait: If True, block until the agent completes. If False, run in background.

        Returns:
            The agent's response text (if wait=True), or a task snapshot (if wait=False).
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
            system_prompt="You are a helpful AI assistant.",
            agent_def=agent_def,
            isolation=isolation,
            name=name,
        )

        if wait and task.status not in ("failed",):
            mgr.wait(task.id, timeout=300)

        if task.status == "completed" and task.result is not None:
            return task.result
        if task.status == "failed":
            return f"Agent failed: {task.error}"
        if task.status == "cancelled":
            return "[Sub-agent was cancelled.]"

        return json.dumps(task.snapshot(), indent=2)


class SendMessageTool(AgentBaseFn):
    """Send a follow-up message to a running sub-agent."""

    @staticmethod
    def static_call(
        target: str,
        message: str,
        **context_variables,
    ) -> str:
        """
        Send a message to a running background agent.

        The message is queued and processed after the agent's current work completes.

        Args:
            target: Agent name or task ID.
            message: The message text to send.

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
    """Create a background task (alias for spawning an agent with wait=False)."""

    @staticmethod
    def static_call(
        prompt: str,
        name: str = "",
        subagent_type: str = "general-purpose",
        **context_variables,
    ) -> str:
        """
        Create a background task.

        Args:
            prompt: Task description.
            name: Optional task name.
            subagent_type: Agent type to use.

        Returns:
            Task snapshot as JSON.
        """
        return AgentTool.static_call(
            prompt=prompt,
            subagent_type=subagent_type,
            name=name,
            wait=False,
        )


class SpawnAgents(AgentBaseFn):
    """Spawn multiple sub-agents in parallel and wait for all to complete."""

    @staticmethod
    def static_call(
        agents: list[dict[str, str]] | str,
        wait: bool = True,
        **context_variables,
    ) -> str:
        """
        Spawn multiple sub-agents in parallel.

        Each agent in the list is a dict with:
            - prompt (str, required): Task description.
            - name (str, optional): Human-readable name.
            - subagent_type (str, optional): Agent type, default "general-purpose".

        Args:
            agents: List of agent specifications (or JSON string).
            wait: If True, block until all agents complete.

        Returns:
            JSON array of results (name, status, result) when wait=True,
            or JSON array of task snapshots when wait=False.
        """
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
                system_prompt="You are a helpful AI assistant.",
                agent_def=agent_def,
                name=spec.get("name", ""),
            )
            tasks.append(task)

        if not wait:
            return json.dumps([t.snapshot() for t in tasks], indent=2, default=str)

        results = []
        for task in tasks:
            mgr.wait(task.id, timeout=300)
            results.append(
                {
                    "name": task.name,
                    "status": task.status,
                    "result": task.result or "",
                    "error": task.error or "",
                }
            )
        return json.dumps(results, indent=2, default=str)


class TaskGetTool(AgentBaseFn):
    """Get the status and result of a task."""

    @staticmethod
    def static_call(task_id: str, **context_variables) -> str:
        """
        Check the status of a background task.

        Args:
            task_id: The task ID or name.

        Returns:
            Task snapshot as JSON.
        """
        mgr = _get_agent_manager()
        task = mgr.tasks.get(task_id) or mgr.get_by_name(task_id)
        if not task:
            return f"Error: task '{task_id}' not found."
        return json.dumps(task.snapshot(), indent=2)


class TaskListTool(AgentBaseFn):
    """List all background tasks."""

    @staticmethod
    def static_call(**context_variables) -> str:
        """
        List all background tasks with their status.

        Returns:
            Formatted task list.
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
    """Get the output of a completed task."""

    @staticmethod
    def static_call(task_id: str, **context_variables) -> str:
        """
        Get the result output of a completed task.

        Args:
            task_id: The task ID or name.

        Returns:
            The task result text, or error message.
        """
        mgr = _get_agent_manager()
        result = mgr.get_result(task_id)
        if result is None:
            task = mgr.get_by_name(task_id)
            if task:
                result = task.result
        return result or f"No output for task '{task_id}' (may still be running)."


class TaskStopTool(AgentBaseFn):
    """Cancel a running task."""

    @staticmethod
    def static_call(task_id: str, **context_variables) -> str:
        """
        Cancel a running background task.

        Args:
            task_id: The task ID or name.

        Returns:
            Confirmation or error message.
        """
        mgr = _get_agent_manager()
        ok = mgr.cancel(task_id)
        return f"Task '{task_id}' cancelled." if ok else f"Could not cancel task '{task_id}'."


class TaskUpdateTool(AgentBaseFn):
    """Send additional instructions to a running task (alias for SendMessage)."""

    @staticmethod
    def static_call(
        task_id: str,
        message: str,
        **context_variables,
    ) -> str:
        """
        Send additional instructions to a running task.

        Args:
            task_id: Task ID or name.
            message: Instructions to send.

        Returns:
            Confirmation or error message.
        """
        return SendMessageTool.static_call(target=task_id, message=message)


_todo_items: list[dict[str, str]] = []


class TodoWriteTool(AgentBaseFn):
    """Structured task list management."""

    @staticmethod
    def static_call(
        todos: str | list[dict[str, str]],
        **context_variables,
    ) -> str:
        """
        Create or update a structured todo list for tracking progress.

        Args:
            todos: Either a JSON string or list of dicts, each with keys:
                ``content`` (task description), ``status`` (pending/in_progress/completed).

        Returns:
            Formatted todo list.
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


class AskUserQuestionTool(AgentBaseFn):
    """Prompt the user for input."""

    @staticmethod
    def static_call(
        question: str,
        **context_variables,
    ) -> str:
        """
        Ask the user a question and return their response.

        Use this when you need clarification or a decision from the user
        before proceeding.

        Args:
            question: The question to ask.

        Returns:
            The user's response text.
        """
        return f"[AskUserQuestion] {question}\n(Waiting for user response — in non-interactive mode, this returns the question itself.)"


class EnterPlanModeTool(AgentBaseFn):
    """Enter plan mode — stops executing and plans instead."""

    @staticmethod
    def static_call(**context_variables) -> str:
        """
        Enter plan mode.

        In plan mode, the agent produces a structured plan without executing
        any actions. Use this when a task is complex and needs a strategy first.

        Returns:
            Confirmation message.
        """
        return "Entered plan mode. Describe your plan without executing actions."


class ExitPlanModeTool(AgentBaseFn):
    """Exit plan mode — resume executing."""

    @staticmethod
    def static_call(**context_variables) -> str:
        """
        Exit plan mode and resume normal execution.

        Returns:
            Confirmation message.
        """
        return "Exited plan mode. Resuming normal execution."


class EnterWorktreeTool(AgentBaseFn):
    """Create and enter an isolated git worktree."""

    @staticmethod
    def static_call(
        branch_name: str = "",
        **context_variables,
    ) -> str:
        """
        Create a git worktree for isolated work.

        Creates a new git worktree branch so changes are isolated from the
        main workspace. Useful for parallel experiments or risky modifications.

        Args:
            branch_name: Branch name for the worktree. Auto-generated if empty.

        Returns:
            Worktree path and branch name, or error.
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
    """Remove a git worktree."""

    @staticmethod
    def static_call(
        worktree_path: str,
        force: bool = False,
        **context_variables,
    ) -> str:
        """
        Remove a git worktree.

        Args:
            worktree_path: Path to the worktree to remove.
            force: Force removal even with uncommitted changes.

        Returns:
            Confirmation or error message.
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
    """Search available tools by keyword."""

    @staticmethod
    def static_call(
        query: str,
        **context_variables,
    ) -> str:
        """
        Search for available tools matching a query.

        Args:
            query: Keyword or description to search for.

        Returns:
            List of matching tools with descriptions.
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
    """Invoke a named skill (reusable prompt template)."""

    @staticmethod
    def static_call(
        skill_name: str,
        args: str = "",
        **context_variables,
    ) -> str:
        """
        Invoke a named skill.

        Skills are reusable prompt templates that can be loaded from
        SKILL.md files or registered programmatically.

        Args:
            skill_name: Name of the skill to invoke.
            args: Optional arguments for the skill.

        Returns:
            The skill's output or an error message.
        """
        try:
            from ..core.paths import xerxes_subdir
            from ..extensions.skills import SkillRegistry

            registry = SkillRegistry()
            skills_dir = xerxes_subdir("skills")
            project_skills = Path.cwd() / "skills"
            registry.discover(str(skills_dir), str(project_skills))

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
    """Edit Jupyter notebook cells."""

    @staticmethod
    def static_call(
        notebook_path: str,
        cell_index: int,
        new_source: str,
        cell_type: str = "code",
        **context_variables,
    ) -> str:
        """
        Edit a cell in a Jupyter notebook (.ipynb file).

        Args:
            notebook_path: Path to the .ipynb file.
            cell_index: Zero-based index of the cell to edit.
            new_source: New source content for the cell.
            cell_type: Cell type (``code`` or ``markdown``).

        Returns:
            Confirmation or error message.
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
    """Language Server Protocol queries."""

    @staticmethod
    def static_call(
        action: str,
        file_path: str = "",
        line: int = 0,
        character: int = 0,
        **context_variables,
    ) -> str:
        """
        Query a Language Server Protocol server.

        Provides code intelligence features like go-to-definition, references,
        and hover information.

        Args:
            action: LSP action — ``definition``, ``references``, ``hover``,
                ``symbols``, ``diagnostics``.
            file_path: Path to the file.
            line: Zero-based line number.
            character: Zero-based character offset.

        Returns:
            LSP query result or error.
        """
        return (
            f"[LSP:{action}] file={file_path} line={line} char={character}\n"
            "LSP tool requires an active language server. In the TUI, this is "
            "handled by the IDE integration layer. Use Grep/Glob for code search instead."
        )


class MCPTool(AgentBaseFn):
    """Invoke an MCP (Model Context Protocol) tool."""

    @staticmethod
    def static_call(
        server_name: str,
        tool_name: str,
        arguments: str | dict | None = None,
        **context_variables,
    ) -> str:
        """
        Call a tool on an MCP server.

        Args:
            server_name: Name of the MCP server.
            tool_name: Tool to invoke on the server.
            arguments: Tool arguments (JSON string or dict).

        Returns:
            Tool result or error.
        """
        try:
            from ..mcp import MCPManager  # noqa: F401

            return (
                f"[MCP] server={server_name} tool={tool_name}\n"
                "Use xerxes.mcp.MCPManager for async MCP tool invocation. "
                "This tool is a placeholder for the synchronous tool interface."
            )
        except ImportError:
            return "Error: xerxes.mcp module not available. Install xerxes[mcp]."


class ListMcpResourcesTool(AgentBaseFn):
    """List resources from MCP servers."""

    @staticmethod
    def static_call(server_name: str = "", **context_variables) -> str:
        """
        List available MCP resources.

        Args:
            server_name: Optional server name filter.

        Returns:
            List of MCP resources or guidance.
        """
        return (
            f"[MCP Resources] server={server_name or '(all)'}\n"
            "Use xerxes.mcp.MCPManager.list_resources() for async MCP resource listing."
        )


class ReadMcpResourceTool(AgentBaseFn):
    """Read an MCP resource."""

    @staticmethod
    def static_call(
        server_name: str,
        uri: str,
        **context_variables,
    ) -> str:
        """
        Read a resource from an MCP server.

        Args:
            server_name: MCP server name.
            uri: Resource URI to read.

        Returns:
            Resource content or guidance.
        """
        return (
            f"[MCP Read] server={server_name} uri={uri}\n"
            "Use xerxes.mcp.MCPManager.read_resource() for async MCP resource reading."
        )


class RemoteTriggerTool(AgentBaseFn):
    """Trigger a remote agent execution."""

    @staticmethod
    def static_call(
        trigger_name: str,
        payload: str = "",
        **context_variables,
    ) -> str:
        """
        Trigger a remote agent to execute.

        Args:
            trigger_name: Name of the remote trigger.
            payload: Optional payload/prompt for the trigger.

        Returns:
            Trigger confirmation or error.
        """
        return f"[RemoteTrigger] name={trigger_name} payload={payload[:100]}\nRemote triggers require configured remote endpoints."


class ScheduleCronTool(AgentBaseFn):
    """Schedule a recurring agent task."""

    @staticmethod
    def static_call(
        schedule: str,
        prompt: str,
        name: str = "",
        **context_variables,
    ) -> str:
        """
        Schedule a recurring agent task on a cron schedule.

        Args:
            schedule: Cron expression (e.g. ``0 9 * * *`` for daily at 9am).
            prompt: The prompt to execute on each run.
            name: Optional name for the scheduled task.

        Returns:
            Schedule confirmation or error.
        """
        return (
            f"[ScheduleCron] schedule={schedule} name={name or '(unnamed)'}\n"
            f"Prompt: {prompt[:100]}\n"
            "Cron scheduling requires a persistent scheduler service."
        )


class HandoffTool(AgentBaseFn):
    """Hand off the current task to a specialized agent.

    Unlike AgentTool (which runs a sub-agent and returns the result),
    HandoffTool transfers the conversation context to another agent type.
    The receiving agent gets a summary of what's been done so far and
    continues the work autonomously.
    """

    @staticmethod
    def static_call(
        target_agent: str,
        reason: str,
        context_summary: str = "",
        prompt: str = "",
        **context_variables,
    ) -> str:
        """
        Hand off the current task to a specialized agent.

        The receiving agent inherits the current provider config and gets
        a summary of what has been done. This is an explicit delegation —
        the parent agent yields control to the specialist.

        Args:
            target_agent: Agent type to hand off to (e.g. ``coder``, ``reviewer``).
            reason: Why this handoff is happening.
            context_summary: Summary of work done so far for the receiving agent.
            prompt: Specific instructions for the receiving agent.

        Returns:
            The receiving agent's response.
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

        mgr.wait(task.id, timeout=300)

        if task.status == "completed" and task.result:
            return task.result
        if task.status == "failed":
            return f"Handoff to {target_agent} failed: {task.error}"
        return json.dumps(task.snapshot(), indent=2)


class PlanTool(AgentBaseFn):
    """Create and execute a multi-step plan using specialized agents.

    The planner breaks down a complex objective into steps, assigns each
    step to the most appropriate agent type, and executes them respecting
    dependency order. Parallel-safe steps run concurrently.
    """

    @staticmethod
    def static_call(
        objective: str,
        execute: bool = True,
        **context_variables,
    ) -> str:
        """
        Create an execution plan for a complex objective.

        Uses an LLM to decompose the objective into discrete steps,
        each assigned to a specialized agent. Steps with no dependencies
        on each other run in parallel.

        Args:
            objective: The high-level objective to plan and execute.
            execute: If True, execute the plan immediately. If False,
                only return the plan without executing.

        Returns:
            The plan (and execution results if execute=True).
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
                mgr.wait(task.id, timeout=300)
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
    """Parse the <plan> XML into a list of step dicts."""
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
