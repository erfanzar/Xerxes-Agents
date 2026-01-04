# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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


"""Dynamic task creator agent for generating tasks from prompts.

This module provides the TaskCreator class for automatically generating
structured task breakdowns from natural language objectives. It uses an
LLM-powered agent to analyze objectives and produce detailed, actionable
task plans that can be executed within the Cortex framework.

The module includes:
- TaskCreator: Main class for generating tasks from prompts
- TaskDefinition: Data class representing individual task specifications
- TaskCreationPlan: Container for complete task breakdowns

Key features:
- Natural language to structured task conversion
- Automatic agent assignment based on task requirements
- Dependency detection and ordering
- Priority and importance scoring
- Tool requirement identification
- Background/approach context integration
- Streaming support for task creation
- Fallback handling for parsing failures

Typical usage example:
    from calute.cortex.task_creator import TaskCreator
    from calute.cortex.agent import CortexAgent

    creator = TaskCreator(verbose=True, model="gpt-4")

    agents = [
        CortexAgent(role="Researcher", goal="Research topics", backstory="Expert"),
        CortexAgent(role="Writer", goal="Write content", backstory="Expert")
    ]

    plan, tasks = creator.create_tasks_from_prompt(
        prompt="Write a research report on AI trends",
        background="Focus on recent developments",
        available_agents=agents
    )

    for task in tasks:
        print(f"Task: {task.description}")
"""

from __future__ import annotations

import re
import typing
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Thread
from typing import Any

from calute.llms.base import BaseLLM

from ..loggings import get_logger
from ..streamer_buffer import StreamerBuffer
from .agent import CortexAgent
from .task import CortexTask
from .templates import PromptTemplate

if typing.TYPE_CHECKING:
    from calute.cortex.cortex import Cortex
    from calute.cortex.enums import ProcessType


@dataclass
class TaskDefinition:
    """Definition of a task to be created from an LLM-generated plan.

    TaskDefinition represents the specification of a single task as parsed
    from the LLM's XML task plan output. It contains all the metadata needed
    to create an actual CortexTask instance.

    Attributes:
        task_id: Unique integer identifier for this task within the plan.
        description: Detailed description of what the task should accomplish.
        expected_output: Description of what successful completion looks like.
        agent_role: Suggested role name for the agent to execute this task.
        dependencies: List of task_ids that must complete before this task.
        context_needed: Whether this task requires context from previous tasks.
        tools_needed: List of tool names that may be needed for execution.
        importance: Priority score from 0.1 (low) to 1.0 (critical).
        validation_required: Whether output should be validated against a schema.
        human_feedback: Whether to request human review of the output.
    """

    task_id: int
    description: str
    expected_output: str
    agent_role: str | None = None
    dependencies: list[int] = field(default_factory=list)
    context_needed: bool = False
    tools_needed: list[str] = field(default_factory=list)
    importance: float = 0.5
    validation_required: bool = False
    human_feedback: bool = False

    def __str__(self) -> str:
        """Return a string representation of the task definition.

        Returns:
            Formatted string showing task ID and truncated description.
        """
        return f"Task {self.task_id}: {self.description[:50]}..."


@dataclass
class TaskCreationPlan:
    """Complete task creation plan containing all task definitions.

    TaskCreationPlan represents the full breakdown of an objective into
    structured tasks. It includes metadata about the plan itself and
    a collection of TaskDefinition instances.

    Attributes:
        plan_id: Unique identifier for this plan (typically based on objective hash).
        objective: The original objective or goal that was broken down.
        approach: Description of the strategy used for task decomposition.
        tasks: List of TaskDefinition instances in execution order.
        estimated_complexity: Overall complexity rating ('simple', 'medium', 'complex').
        total_tasks: Current count of tasks in the plan.
        sequential: Whether tasks should be executed in sequence (True) or
            can be parallelized where dependencies allow (False).
    """

    plan_id: str
    objective: str
    approach: str
    tasks: list[TaskDefinition] = field(default_factory=list)
    estimated_complexity: str = "medium"
    total_tasks: int = 0
    sequential: bool = True

    def add_task(self, task: TaskDefinition):
        """Add a task to the plan and update the task count.

        Args:
            task: The TaskDefinition to add to the plan.

        Side Effects:
            Appends task to the tasks list and updates total_tasks.
        """
        self.tasks.append(task)
        self.total_tasks = len(self.tasks)

    def get_task_by_id(self, task_id: int) -> TaskDefinition | None:
        """Retrieve a task definition by its ID.

        Args:
            task_id: The unique identifier of the task to find.

        Returns:
            The matching TaskDefinition, or None if not found.
        """
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None


class TaskCreator:
    """Dynamic task creator that generates structured tasks from natural language prompts.

    TaskCreator uses an LLM-powered agent to analyze high-level objectives and
    automatically generate detailed, actionable task breakdowns. It produces
    structured XML output that is parsed into TaskDefinition and TaskCreationPlan
    objects, which can then be converted to CortexTask instances for execution.

    Attributes:
        verbose: Whether to output detailed logging during task creation.
        model: Model identifier for the creator agent's LLM.
        llm: Direct BaseLLM instance for LLM operations.
        max_tasks: Maximum number of tasks to include in a plan.
        auto_assign_agents: Whether to suggest agent assignments based on roles.
        logger: Logger instance for verbose output (None if verbose=False).
        template_engine: PromptTemplate engine for rendering prompts.
        creator_agent: Internal CortexAgent that performs task creation.

    Class Attributes:
        TASK_CREATION_TEMPLATE: Jinja2 template for generating task creation prompts.
            Includes placeholders for objective, background, available agents,
            and constraints.

    Example:
        >>> creator = TaskCreator(model="gpt-4", verbose=True)
        >>> plan, tasks = creator.create_tasks_from_prompt(
        ...     prompt="Build a REST API for user management",
        ...     background="Use FastAPI with PostgreSQL",
        ...     available_agents=my_agents
        ... )
        >>> print(f"Created {len(tasks)} tasks")
    """

    TASK_CREATION_TEMPLATE = """
You are a task creation specialist. Create a detailed set of tasks for the following objective.

OBJECTIVE: {{ objective }}

{% if background %}
BACKGROUND/APPROACH:
{{ background }}
This background should guide your approach to breaking down the tasks.
{% else %}
BACKGROUND/APPROACH: Use your best judgment to determine the optimal approach.
{% endif %}

{% if available_agents %}
AVAILABLE AGENTS:
{% for agent in available_agents %}
- {{ agent.role }}: {{ agent.goal }}
{% endfor %}
{% endif %}

{% if constraints %}
CONSTRAINTS:
{{ constraints }}
{% endif %}

Create a task breakdown using the following XML format:

<task_plan>
    <objective>{{ objective }}</objective>
    <approach>Brief description of the approach taken based on background</approach>
    <complexity>simple|medium|complex</complexity>
    <sequential>true|false</sequential>

    <task id="1">
        <description>Clear description of what needs to be done</description>
        <expected_output>What the successful completion looks like</expected_output>
        <agent_role>Optional: Suggested agent role for this task</agent_role>
        <dependencies></dependencies>
        <context_needed>true|false</context_needed>
        <tools_needed>tool1,tool2</tools_needed>
        <importance>0.1-1.0</importance>
        <validation_required>true|false</validation_required>
        <human_feedback>true|false</human_feedback>
    </task>

    <task id="2">
        <description>Another task description</description>
        <expected_output>Expected result</expected_output>
        <agent_role>Another Agent Role</agent_role>
        <dependencies>1</dependencies>
        <context_needed>true</context_needed>
        <tools_needed></tools_needed>
        <importance>0.5</importance>
        <validation_required>false</validation_required>
        <human_feedback>false</human_feedback>
    </task>
</task_plan>

INSTRUCTIONS:
1. Break down the objective into clear, actionable tasks
2. Each task should be self-contained but can depend on others
3. Consider the background/approach when determining task breakdown
4. Assign importance scores (0.1=low, 0.5=medium, 1.0=critical)
5. Specify if tasks need context from previous tasks
6. Identify any tools or capabilities needed
7. Mark tasks that need validation or human feedback
8. Create between 2-10 tasks as appropriate

Respond ONLY with the XML plan, no additional text.
"""

    def __init__(
        self,
        verbose: bool = True,
        model: str | None = None,
        llm: BaseLLM | None = None,
        max_tasks: int = 10,
        auto_assign_agents: bool = True,
    ):
        """Initialize the TaskCreator with configuration options.

        Creates a TaskCreator instance with an internal CortexAgent specialized
        for task breakdown and planning. The agent uses the provided model
        or LLM to generate structured task plans.

        Args:
            verbose: Whether to output detailed logging during task creation.
                Enables informative log messages about the creation process.
            model: Model identifier string for the creator agent (e.g., 'gpt-4').
                Used if llm is not provided.
            llm: Direct BaseLLM instance to use for task creation. Takes
                precedence over the model parameter if both are provided.
            max_tasks: Maximum number of tasks to include in any plan.
                Plans with more tasks will be truncated (default: 10).
            auto_assign_agents: Whether to automatically create CortexTask
                instances with agent assignments when available_agents
                is provided to create_tasks_from_prompt().

        Side Effects:
            - Creates internal creator_agent CortexAgent instance
            - Initializes logger if verbose=True
            - Sets up template_engine for prompt rendering
        """
        self.verbose = verbose
        self.model = model
        self.llm = llm
        self.max_tasks = max_tasks
        self.auto_assign_agents = auto_assign_agents
        self.logger = get_logger() if verbose else None
        self.template_engine = PromptTemplate()

        self.creator_agent = CortexAgent(
            role="Task Creation Specialist",
            goal="Break down complex objectives into well-structured, actionable tasks",
            backstory="""You are an expert at analyzing objectives and creating detailed task breakdowns.
            You understand how to decompose complex goals into manageable steps, identify dependencies,
            and structure work for optimal execution. You consider the provided background/approach
            to tailor your task creation strategy.""",
            model=model,
            llm=llm,
            verbose=verbose,
            allow_delegation=False,
        )

        self.template_engine.env.from_string(self.TASK_CREATION_TEMPLATE)

    def create_tasks_from_prompt(
        self,
        prompt: str,
        background: str | None = None,
        available_agents: list[CortexAgent] | None = None,
        constraints: str | None = None,
        stream: bool = False,
        stream_callback: Callable[[Any], None] | None = None,
        streamer_buffer: StreamerBuffer | None = None,
    ) -> tuple[TaskCreationPlan, list[CortexTask] | None]:
        """Create structured tasks from a natural language prompt.

        Analyzes the provided objective and generates a detailed task breakdown
        using the internal creator agent. Optionally assigns agents to tasks
        based on role matching.

        Args:
            prompt: The objective or goal to break down into tasks.
            background: Optional approach or context to guide task creation.
                Helps the creator understand the preferred methodology.
            available_agents: Optional list of CortexAgent instances for
                automatic task assignment. If provided and auto_assign_agents
                is True, returns CortexTask instances with agents assigned.
            constraints: Optional constraints or requirements that tasks
                must satisfy.
            stream: Whether to stream the LLM response during creation.
            stream_callback: Optional callback invoked with each streamed chunk.
            streamer_buffer: Optional StreamerBuffer for collecting stream output.

        Returns:
            Tuple of (TaskCreationPlan, list[CortexTask] | None):
            - TaskCreationPlan containing all parsed TaskDefinition objects
            - List of CortexTask instances if available_agents provided and
              auto_assign_agents is True, otherwise None

        Note:
            If XML parsing fails, a fallback single-task plan is returned.
        """
        if self.verbose and self.logger:
            self.logger.info(f"📝 Creating tasks for: {prompt[:100]}...")
            if background:
                self.logger.info(f"📋 Using approach: {background[:100]}...")

        creation_prompt = self.template_engine.render(
            self.TASK_CREATION_TEMPLATE,
            objective=prompt,
            background=background,
            available_agents=available_agents,
            constraints=constraints,
        )

        try:
            if stream:
                response = self.creator_agent.execute(
                    task_description=creation_prompt,
                    streamer_buffer=streamer_buffer,
                    stream_callback=stream_callback,
                )
            else:
                response = self.creator_agent.execute(task_description=creation_prompt)

            task_plan = self._parse_xml_tasks(response, prompt)

            if self.verbose:
                self._log_task_summary(task_plan)

            if available_agents and self.auto_assign_agents:
                cortex_tasks = self._create_cortex_tasks(task_plan, available_agents)
                return task_plan, cortex_tasks

            return task_plan, None

        except Exception as e:
            if self.verbose and self.logger:
                self.logger.error(f"❌ Failed to create tasks: {e}")

            return self._create_fallback_plan(prompt, background)

    def _parse_xml_tasks(self, xml_response: str, objective: str) -> TaskCreationPlan:
        """Parse XML task response into a TaskCreationPlan object.

        Extracts the task_plan XML structure from the LLM response and
        parses it into TaskDefinition objects within a TaskCreationPlan.

        Args:
            xml_response: Raw LLM output containing XML task plan.
            objective: Original objective for fallback and plan identification.

        Returns:
            TaskCreationPlan with parsed tasks, truncated to max_tasks if needed.

        Raises:
            ValueError: If XML parsing fails with details about the error.
        """
        try:
            xml_match = re.search(r"<task_plan>.*?</task_plan>", xml_response, re.DOTALL)
            if xml_match:
                xml_content = xml_match.group(0)
            else:
                xml_content = xml_response

            root = ET.fromstring(xml_content)

            plan = TaskCreationPlan(
                plan_id=f"plan_{hash(objective) % 10000}",
                objective=root.find("objective").text or objective,
                approach=root.find("approach").text or "Standard approach",
                estimated_complexity=root.find("complexity").text or "medium",
                sequential=root.find("sequential").text == "true" if root.find("sequential") is not None else True,
            )

            for task_elem in root.findall("task"):
                task_id = int(task_elem.get("id"))

                dependencies = []
                deps_elem = task_elem.find("dependencies")
                if deps_elem is not None and deps_elem.text:
                    deps_text = deps_elem.text.strip()
                    if deps_text:
                        dependencies = [int(x.strip()) for x in deps_text.split(",")]

                tools_needed = []
                tools_elem = task_elem.find("tools_needed")
                if tools_elem is not None and tools_elem.text:
                    tools_text = tools_elem.text.strip()
                    if tools_text:
                        tools_needed = [tool.strip() for tool in tools_text.split(",")]

                importance = 0.5
                importance_elem = task_elem.find("importance")
                if importance_elem is not None and importance_elem.text:
                    try:
                        importance = float(importance_elem.text)
                    except ValueError:
                        importance = 0.5

                task_def = TaskDefinition(
                    task_id=task_id,
                    description=task_elem.find("description").text or "",
                    expected_output=task_elem.find("expected_output").text or "",
                    agent_role=task_elem.find("agent_role").text if task_elem.find("agent_role") is not None else None,
                    dependencies=dependencies,
                    context_needed=task_elem.find("context_needed").text == "true"
                    if task_elem.find("context_needed") is not None
                    else False,
                    tools_needed=tools_needed,
                    importance=importance,
                    validation_required=task_elem.find("validation_required").text == "true"
                    if task_elem.find("validation_required") is not None
                    else False,
                    human_feedback=task_elem.find("human_feedback").text == "true"
                    if task_elem.find("human_feedback") is not None
                    else False,
                )

                plan.add_task(task_def)

            if len(plan.tasks) > self.max_tasks:
                plan.tasks = plan.tasks[: self.max_tasks]
                plan.total_tasks = self.max_tasks

            return plan

        except Exception as e:
            if self.verbose and self.logger:
                self.logger.error(f"❌ Failed to parse XML tasks: {e}")
            raise ValueError(f"Invalid XML task format: {e}") from e

    def _create_cortex_tasks(self, task_plan: TaskCreationPlan, available_agents: list[CortexAgent]) -> list[CortexTask]:
        """Convert TaskDefinitions to executable CortexTask objects.

        Maps TaskDefinition instances to CortexTask instances with agent
        assignments based on role matching. If no matching agent is found,
        the first available agent is used as a fallback.

        Args:
            task_plan: The TaskCreationPlan containing TaskDefinitions.
            available_agents: List of CortexAgent instances for assignment.

        Returns:
            List of CortexTask instances with agents assigned and
            dependencies linked to previously created tasks.
        """
        cortex_tasks = []
        agent_map = {agent.role: agent for agent in available_agents}

        for task_def in task_plan.tasks:
            agent = None
            if task_def.agent_role and task_def.agent_role in agent_map:
                agent = agent_map[task_def.agent_role]
            elif available_agents:
                agent = available_agents[0]

            dependencies = [
                cortex_tasks[dep_id - 1] for dep_id in task_def.dependencies
                if dep_id > 0 and dep_id - 1 < len(cortex_tasks)
            ]

            cortex_task = CortexTask(
                description=task_def.description,
                expected_output=task_def.expected_output,
                agent=agent,
                importance=task_def.importance,
                human_feedback=task_def.human_feedback,
                context=dependencies if dependencies else (True if task_def.context_needed else None),
                dependencies=dependencies,
            )

            cortex_tasks.append(cortex_task)

        return cortex_tasks

    def _create_fallback_plan(self, objective: str, background: str | None) -> tuple[TaskCreationPlan, None]:
        """Create a simple fallback plan when XML parsing fails.

        Generates a minimal single-task plan that wraps the original
        objective as one task. Used as a graceful degradation when
        the LLM response cannot be parsed.

        Args:
            objective: The original objective to wrap as a single task.
            background: Optional background to use as approach description.

        Returns:
            Tuple of (TaskCreationPlan, None) with a single task.
        """
        plan = TaskCreationPlan(
            plan_id=f"fallback_{hash(objective) % 10000}",
            objective=objective,
            approach=background or "Simple execution",
            estimated_complexity="simple",
        )

        task = TaskDefinition(
            task_id=1,
            description=f"Execute the objective: {objective}",
            expected_output="Complete the objective successfully",
            importance=1.0,
        )
        plan.add_task(task)

        return plan, None

    def _log_task_summary(self, plan: TaskCreationPlan):
        """Log a formatted summary of the created task plan.

        Outputs detailed information about the plan including objective,
        approach, complexity, and individual task descriptions with
        their dependencies and agent assignments.

        Args:
            plan: The TaskCreationPlan to summarize.

        Note:
            Only logs if self.logger is not None (verbose mode enabled).
        """
        if self.logger:
            self.logger.info("📋 Task Creation Summary:")
            self.logger.info(f"  • Objective: {plan.objective}")
            self.logger.info(f"  • Approach: {plan.approach}")
            self.logger.info(f"  • Total tasks: {plan.total_tasks}")
            self.logger.info(f"  • Complexity: {plan.estimated_complexity}")
            self.logger.info(f"  • Sequential: {plan.sequential}")

            for task in plan.tasks:
                deps = f" (deps: {task.dependencies})" if task.dependencies else ""
                agent = f" -> {task.agent_role}" if task.agent_role else ""
                self.logger.info(f"    {task.task_id}. {task.description[:50]}...{agent}{deps}")

    def create_and_execute(
        self,
        prompt: str,
        background: str | None,
        cortex: Cortex,
        process_type: ProcessType = None,
        use_streaming: bool = False,
        stream_callback: Callable[[Any], None] | None = None,
        log_process: bool = False,
    ) -> Any | tuple[StreamerBuffer, Thread]:
        """Create tasks from a prompt and immediately execute them using a Cortex.

        Combines task creation and execution into a single operation. First
        generates tasks from the prompt using available agents in the Cortex,
        then kicks off execution of those tasks.

        Args:
            prompt: The objective or goal to accomplish.
            background: Optional approach or context for task creation.
            cortex: The Cortex instance containing agents for execution.
                Must have agents defined.
            process_type: Optional ProcessType to override the Cortex's
                default process type for this execution.
            use_streaming: Whether to stream execution output.
            stream_callback: Optional callback for streaming chunks.
            log_process: Whether to log the execution process.

        Returns:
            If use_streaming=False: The execution result from cortex.kickoff().
            If use_streaming=True: Tuple of (StreamerBuffer, Thread).

        Raises:
            ValueError: If the Cortex has no agents defined.

        Note:
            The original process_type is restored after execution if overridden.
        """

        if cortex.agents:
            _task_plan, cortex_tasks = self.create_tasks_from_prompt(
                prompt=prompt,
                background=background,
                available_agents=cortex.agents,
            )
        else:
            raise ValueError("Cortex must have agents defined")

        cortex.tasks = cortex_tasks

        if process_type:
            original_process = cortex.process
            cortex.process = process_type

        if use_streaming:
            buffer, thread = cortex.kickoff(use_streaming=True, stream_callback=stream_callback, log_process=log_process)
        else:
            result = cortex.kickoff(use_streaming=False, stream_callback=stream_callback, log_process=log_process)

        if process_type:
            cortex.process = original_process
        if use_streaming:
            return buffer, thread
        return result

    def create_ui(self, cortex: Cortex | None = None):
        """Launch a UI application for interacting with the task creator.

        Creates and launches an interactive user interface that allows
        users to create and execute tasks visually through the TaskCreator.

        Args:
            cortex: Optional Cortex instance to use for task execution.
                If not provided, only task creation will be available.

        Returns:
            The launched UI application instance.
        """
        from calute.ui import launch_application

        return launch_application(executor=self, agent=cortex)
