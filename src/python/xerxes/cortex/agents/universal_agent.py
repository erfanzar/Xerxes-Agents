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
"""Universal agent and task creator with a broad set of built-in tools."""

from typing import Any, cast

from ...tools import ExecutePythonCode, GoogleSearch, WriteFile
from ...tools.coding_tools import (
    analyze_code_structure,
    apply_diff,
    copy_file,
    create_diff,
    delete_file,
    find_and_replace,
    git_add,
    git_apply_patch,
    git_diff,
    git_log,
    git_status,
    list_directory,
    move_file,
    read_file,
)
from ...types import AgentCapability
from ..core.tool import CortexTool
from ..orchestration.task import CortexTask
from .agent import CortexAgent


class UniversalAgent(CortexAgent):
    """A general-purpose agent pre-configured with common system tools.

    Inherits from ``CortexAgent`` and provides a default role, goal, backstory,
    and a comprehensive tool set including web search, file operations, git
    operations, code analysis, and Python execution.
    """

    def __init__(
        self,
        llm=None,
        verbose: bool = True,
        allow_delegation: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        additional_tools: list[Any] | None = None,
    ):
        """Build a Universal Task Executor with the default tool catalogue.

        All standard :class:`CortexAgent` constructor knobs are forwarded;
        additional tools, when supplied, are appended after the built-in
        ones.
        """

        tools = self._build_tool_set(additional_tools)

        super().__init__(
            role="Universal Task Executor",
            goal="""Execute any type of task efficiently by leveraging a comprehensive set of real, functional tools
            including research, analysis, content generation, code execution, data processing, and system operations""",
            backstory="""You are a highly versatile AI agent with extensive real-world capabilities.
            You have access to actual functional tools that can perform real operations including:
            - Web searching and research
            - File operations (read, write, save, copy, move, delete)
            - Git operations (status, diff, log, add, apply patches)
            - Code analysis and manipulation
            - Python code execution
            - Diff creation and application
            You adapt your approach based on task requirements and always use the most appropriate tools.""",
            instructions="""When calling functions, ALWAYS ensure:
            1. JSON arguments are properly formatted with all required fields having values
            2. For file/directory paths, use "." for current directory if not specified
            3. Never leave a JSON key without a value - use null, empty string "", or appropriate default
            4. Example correct format: {"repo_path": ".", "file_path": null, "staged": false}
            5. Example WRONG format: {"repo_path": "file_path": null} (missing value for repo_path)

            Always double-check your JSON before submitting function calls.""",
            tools=tools,
            llm=llm,
            verbose=verbose,
            allow_delegation=allow_delegation,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.capabilities = cast(list[AgentCapability], self._define_capabilities())

    def _build_tool_set(self, additional_tools: list[Any] | None = None) -> list[Any]:
        """Return the built-in tool list plus any caller-supplied tools."""

        tools = [
            GoogleSearch,
            WriteFile,
            CortexTool.from_function(
                read_file,
                name="read_file",
                description="Read file with line numbers and range selection",
            ),
            CortexTool.from_function(
                list_directory,
                name="list_directory",
                description="List directory contents with advanced filtering and recursive options",
            ),
            CortexTool.from_function(
                copy_file,
                name="copy_file",
                description="Copy files or directories with overwrite control",
            ),
            CortexTool.from_function(
                move_file,
                name="move_file",
                description="Move files or directories",
            ),
            CortexTool.from_function(
                delete_file,
                name="delete_file",
                description="Delete files or directories with safety checks",
            ),
            CortexTool.from_function(
                git_status,
                name="git_status",
                description="Get git repository status",
            ),
            CortexTool.from_function(
                git_diff,
                name="git_diff",
                description="Get git diff for changes (staged or unstaged)",
            ),
            CortexTool.from_function(
                git_log,
                name="git_log",
                description="Get git commit history",
            ),
            CortexTool.from_function(
                git_add,
                name="git_add",
                description="Stage files for git commit",
            ),
            CortexTool.from_function(
                git_apply_patch,
                name="git_apply_patch",
                description="Apply a git patch",
            ),
            CortexTool.from_function(
                create_diff,
                name="create_diff",
                description="Create unified diff between two text contents",
            ),
            CortexTool.from_function(
                apply_diff,
                name="apply_diff",
                description="Apply a unified diff to original content",
            ),
            CortexTool.from_function(
                find_and_replace,
                name="find_and_replace",
                description="Find and replace text in files with regex support",
            ),
            CortexTool.from_function(
                analyze_code_structure,
                name="analyze_code_structure",
                description="Analyze code file structure (classes, functions, imports)",
            ),
            ExecutePythonCode,
        ]

        if additional_tools:
            tools.extend(additional_tools)

        return tools

    def _define_capabilities(self) -> list[str]:
        """Return the human-readable capability list for introspection."""

        return [
            "Web research and information gathering",
            "Advanced file operations (read, write, copy, move, delete)",
            "Directory listing with filtering and recursive options",
            "Git operations (status, diff, log, add, apply patches)",
            "Code diff creation and application",
            "Find and replace with regex support",
            "Code structure analysis (classes, functions, imports)",
            "Python code execution and testing",
            "Content generation and saving",
        ]

    def describe_capabilities(self) -> str:
        """Return a printable capability summary and tool count."""

        cap_list = "\n".join([f"• {cap}" for cap in self.capabilities])
        return f"""Universal Agent Capabilities:

{cap_list}

Total Tools Available: {len(self.tools)}
"""


class UniversalTaskCreator:
    """Creates tasks and assigns them to a universal agent or specialized agents.

    Uses a ``TaskCreator`` to decompose prompts into structured tasks, then
    maps them to the most appropriate agent from a provided list.
    """

    def __init__(
        self,
        llm=None,
        verbose: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """Construct a :class:`TaskCreator` paired with a :class:`UniversalAgent`.

        The same ``llm`` instance backs both — the task creator uses it for
        decomposition and the universal agent uses it as the fallback
        executor when no specialised agent matches.
        """

        from ..orchestration.task_creator import TaskCreator

        self.task_creator = TaskCreator(
            verbose=verbose,
            llm=llm,
            auto_assign_agents=True,
        )

        self.universal_agent = UniversalAgent(
            llm=llm,
            verbose=verbose,
            allow_delegation=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.llm = llm
        self.verbose = verbose

    def create_and_assign_tasks(
        self,
        prompt: str,
        background: str | None = None,
        specialized_agents: list[CortexAgent] | None = None,
    ) -> tuple:
        """Decompose ``prompt`` into tasks and route each one to an agent.

        Tasks whose suggested role substring-matches a specialised agent are
        bound to that agent; the rest fall back to the universal agent.

        Returns:
            ``(TaskCreationPlan, list[CortexTask])`` when specialised
            agents are present so the caller can run the tasks directly;
            otherwise the raw :class:`TaskCreator` result.
        """

        all_agents = []
        if specialized_agents:
            all_agents.extend(specialized_agents)
        all_agents.append(self.universal_agent)
        result = self.task_creator.create_tasks_from_prompt(
            prompt=prompt,
            background=background,
            available_agents=all_agents,
        )

        if not isinstance(result, tuple):
            cortex_tasks = []
            for task_def in result.tasks:
                assigned_agent = self.universal_agent
                if task_def.agent_role and specialized_agents:
                    for agent in specialized_agents:
                        if (
                            agent.role.lower() in task_def.agent_role.lower()
                            or task_def.agent_role.lower() in agent.role.lower()
                        ):
                            assigned_agent = agent
                            break

                cortex_task = CortexTask(
                    description=task_def.description,
                    expected_output=task_def.expected_output,
                    agent=assigned_agent,
                    importance=task_def.importance,
                    human_feedback=task_def.human_feedback,
                )
                cortex_tasks.append(cortex_task)

            return result, cortex_tasks

        return result
