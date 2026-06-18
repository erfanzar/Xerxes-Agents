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
"""Workflow tools: todo tracking, user questions, plan/worktree modes."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from ...types import AgentBaseFn
from .agent_ops import _get_agent_manager, _subagent_wait_timeout

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
            raise RuntimeError("AskUserQuestion callback was never registered; daemon bootstrap is broken")
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
        from ...runtime.config_context import emit_event, get_config, set_config

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
        from ...runtime.bridge import populate_registry

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
            from ...core.paths import xerxes_subdir
            from ...extensions.skills import SkillRegistry

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
        from ...agents.definitions import get_agent_definition, list_agent_definitions
        from ...runtime.config_context import emit_event, get_inheritable

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

        from ...streaming.events import AgentState, TextChunk
        from ...streaming.loop import run

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
