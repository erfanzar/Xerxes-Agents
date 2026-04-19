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


"""Universal Agent for Cortex framework.

This module provides the UniversalAgent class, a versatile agent that can handle
any type of task with comprehensive real tools. It extends CortexAgent with
pre-configured tools for web research, file operations, git operations,
code analysis, and Python execution.

The module also provides UniversalTaskCreator for creating and assigning tasks
using the universal agent as a fallback for unassigned tasks.

Key features:
- Web searching and research via DuckDuckGo
- File operations (read, write, copy, move, delete)
- Git operations (status, diff, log, add, apply patches)
- Code analysis and manipulation
- Python code execution
- Diff creation and application

Typical usage example:
    agent = UniversalAgent(
        llm=my_llm,
        verbose=True,
        allow_delegation=True
    )
    result = agent.execute("Analyze the project structure")


    task_creator = UniversalTaskCreator(llm=my_llm)
    task_plan, tasks = task_creator.create_and_assign_tasks(
        prompt="Build a REST API",
        specialized_agents=[api_expert_agent]
    )
"""

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
    """Versatile agent that can handle any type of task with comprehensive tools.

    UniversalAgent extends CortexAgent with a pre-configured set of real,
    functional tools enabling it to perform a wide variety of tasks including
    web research, file operations, git management, code analysis, and Python
    code execution. It serves as an all-purpose agent suitable for general
    tasks or as a fallback when specialized agents are not available.

    The agent comes pre-configured with:
    - GoogleSearch: Web searching and information gathering
    - WriteFile: File writing capabilities
    - read_file: Read files with line numbers and range selection
    - list_directory: Directory listing with filtering options
    - copy_file, move_file, delete_file: File management operations
    - git_status, git_diff, git_log, git_add, git_apply_patch: Git operations
    - create_diff, apply_diff: Diff creation and application
    - find_and_replace: Text replacement with regex support
    - analyze_code_structure: Code structure analysis
    - ExecutePythonCode: Python code execution

    Attributes:
        capabilities: List of string descriptions of the agent's capabilities.

    Example:
        agent = UniversalAgent(
            llm=llm_instance,
            verbose=True,
            additional_tools=[my_custom_tool]
        )
        result = agent.execute("Search for information about Python")
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
        """Initialize the Universal Agent with comprehensive real tools.

        Creates a new UniversalAgent instance with a complete set of pre-configured
        tools for handling various task types. The agent is initialized with
        sensible defaults and can be customized through the provided parameters.

        Args:
            llm: Optional LLM instance to use for language model operations.
                If None, will use the default LLM configuration.
            verbose: Whether to output detailed logs during execution.
                Defaults to True.
            allow_delegation: Whether the agent can delegate tasks to other
                agents when operating in a multi-agent system. Defaults to True.
            temperature: LLM temperature setting controlling response randomness.
                Values range from 0.0 (deterministic) to 1.0 (creative).
                Defaults to 0.7.
            max_tokens: Maximum number of tokens for LLM responses.
                Defaults to 4096.
            additional_tools: Optional list of additional tools to include
                alongside the built-in tool set. These will be appended to
                the default tools.

        Side Effects:
            - Initializes parent CortexAgent with generated configuration
            - Builds the comprehensive tool set
            - Defines agent capabilities based on available tools
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
        """Build the comprehensive set of real, functional tools.

        Assembles the complete tool set for the universal agent, including
        web search, file operations, git operations, code analysis, and
        Python execution tools. Additional custom tools can be appended.

        Args:
            additional_tools: Optional list of additional tools to include.
                These tools will be appended to the default tool set.

        Returns:
            List of tool instances ready for use by the agent. Includes
            both class-based tools (GoogleSearch, WriteFile, ExecutePythonCode)
            and function-based tools wrapped with CortexTool.from_function().
        """
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
        """Define the agent's capabilities based on available tools.

        Generates a human-readable list of capability descriptions that
        describe what the agent can do. These descriptions are based on
        the tools configured for the agent.

        Returns:
            List of capability description strings, each describing a
            category of tasks the agent can perform (e.g., "Web research
            and information gathering", "Git operations").
        """
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
        """Get a formatted description of the agent's capabilities.

        Generates a human-readable, formatted string describing all
        capabilities of the universal agent, including a bulleted list
        of capabilities and the total number of available tools.

        Returns:
            Formatted multi-line string containing the agent's capabilities
            as a bulleted list, followed by the total tool count.

        Example:
            >>> agent = UniversalAgent()
            >>> print(agent.describe_capabilities())
            Universal Agent Capabilities:

            • Web research and information gathering
            • Advanced file operations (read, write, copy, move, delete)
            ...

            Total Tools Available: 16
        """
        cap_list = "\n".join([f"• {cap}" for cap in self.capabilities])
        return f"""Universal Agent Capabilities:

{cap_list}

Total Tools Available: {len(self.tools)}
"""


class UniversalTaskCreator:
    """Task creator that uses UniversalAgent as default for unassigned tasks.

    UniversalTaskCreator wraps a TaskCreator and UniversalAgent to provide
    automatic task creation and assignment. When specialized agents are provided,
    it attempts to match tasks to appropriate agents based on role similarity.
    Tasks that cannot be matched to a specialized agent are automatically
    assigned to the built-in UniversalAgent.

    This class simplifies the process of creating and executing multi-step
    task plans by handling agent assignment automatically.

    Attributes:
        task_creator: Internal TaskCreator instance for generating task plans.
        universal_agent: UniversalAgent instance used for unassigned tasks.
        llm: LLM instance used by both the task creator and universal agent.
        verbose: Whether to output detailed logs.

    Example:
        creator = UniversalTaskCreator(llm=my_llm, verbose=True)
        task_plan, cortex_tasks = creator.create_and_assign_tasks(
            prompt="Build a user authentication system",
            specialized_agents=[security_agent, database_agent]
        )



    """

    def __init__(
        self,
        llm=None,
        verbose: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """Initialize UniversalTaskCreator with a universal agent.

        Creates a new UniversalTaskCreator instance with an internal TaskCreator
        and UniversalAgent configured with the provided parameters.

        Args:
            llm: Optional LLM instance to use for both task creation and
                the universal agent. If None, uses default configuration.
            verbose: Whether to output detailed logs during operation.
                Defaults to True.
            temperature: LLM temperature setting for the universal agent.
                Values range from 0.0 (deterministic) to 1.0 (creative).
                Defaults to 0.7.
            max_tokens: Maximum number of tokens for universal agent responses.
                Defaults to 4096.

        Side Effects:
            - Creates internal TaskCreator with auto_assign_agents enabled
            - Creates UniversalAgent with the specified configuration
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
        """Create tasks and assign them, using universal agent for unassigned tasks.

        Generates a task plan from the given prompt and assigns each task to
        an appropriate agent. Tasks are matched to specialized agents based on
        role name similarity (case-insensitive substring matching). Any tasks
        that cannot be matched to a specialized agent are assigned to the
        built-in UniversalAgent.

        Args:
            prompt: The objective or goal to create tasks for. This is passed
                to the task creator to generate a structured task plan.
            background: Optional background information or context to help
                guide task creation and provide additional constraints.
            specialized_agents: Optional list of CortexAgent instances with
                specific roles. Tasks will be matched to these agents based
                on role name similarity before falling back to universal agent.

        Returns:
            Tuple containing (task_plan, cortex_tasks) where:
            - task_plan: The original TaskPlan generated by the task creator
            - cortex_tasks: List of CortexTask instances with agents assigned

            If the task creator returns a tuple directly, that tuple is
            returned as-is.

        Example:
            task_plan, tasks = creator.create_and_assign_tasks(
                prompt="Create a web scraper",
                background="Should handle JavaScript-rendered pages",
                specialized_agents=[browser_agent, data_agent]
            )
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
