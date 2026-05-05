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
"""Task creation specialist that decomposes objectives into executable tasks."""

from __future__ import annotations

import re
import typing
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Thread
from typing import Any, Literal

from xerxes.llms.base import BaseLLM

from ...core.streamer_buffer import StreamerBuffer
from ...logging.console import get_logger
from ..agents.agent import CortexAgent
from ..core.templates import PromptTemplate
from .task import CortexTask

if typing.TYPE_CHECKING:
    from xerxes.cortex.core.enums import ProcessType
    from xerxes.cortex.orchestration.cortex import Cortex


@dataclass
class TaskDefinition:
    """A declarative description of a single task within a creation plan.

    Attributes:
        task_id (int): Unique identifier for the task.
        description (str): What the task should accomplish.
        expected_output (str): Description of the desired result.
        agent_role (str | None): Suggested agent role for execution.
        dependencies (list[int]): IDs of tasks this task depends on.
        context_needed (bool): Whether outputs from prior tasks are needed.
        tools_needed (list[str]): Names of tools required for the task.
        importance (float): Importance score from 0.1 to 1.0.
        validation_required (bool): Whether the output should be validated.
        human_feedback (bool): Whether human feedback is requested.
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
        """Return a concise description of the task definition.

        Returns:
            str: Formatted as ``"Task {id}: {description}..."``.
                OUT: Suitable for logging and display.
        """

        return f"Task {self.task_id}: {self.description[:50]}..."


@dataclass
class TaskCreationPlan:
    """A plan produced by decomposing an objective into individual tasks.

    Attributes:
        plan_id (str): Unique identifier for the plan.
        objective (str): The original high-level objective.
        approach (str): Description of the chosen decomposition strategy.
        tasks (list[TaskDefinition]): The tasks derived from the objective.
        estimated_complexity (Literal["simple", "medium", "complex"]): Overall complexity.
        total_tasks (int): Total number of tasks in the plan.
        sequential (bool): Whether the tasks are intended to run sequentially.
    """

    plan_id: str
    objective: str
    approach: str
    tasks: list[TaskDefinition] = field(default_factory=list)
    estimated_complexity: Literal["simple", "medium", "complex"] = "medium"
    total_tasks: int = 0
    sequential: bool = True

    def add_task(self, task: TaskDefinition):
        """Append a task definition and update the total count.

        Args:
            task (TaskDefinition): The task to add.
                IN: Appended to ``self.tasks``.
                OUT: ``self.total_tasks`` is updated to reflect the new length.
        """

        self.tasks.append(task)
        self.total_tasks = len(self.tasks)

    def get_task_by_id(self, task_id: int) -> TaskDefinition | None:
        """Retrieve a task definition by its ID.

        Args:
            task_id (int): The task identifier to search for.
                IN: Matched against ``task.task_id``.
                OUT: Used for lookup in ``self.tasks``.

        Returns:
            TaskDefinition | None: The matching task, or ``None`` if not found.
                OUT: Direct reference from the task list.
        """

        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None


class TaskCreator:
    """Uses an LLM agent to break down objectives into structured task plans.

    The ``TaskCreator`` renders a prompt template, invokes a ``CortexAgent``,
    parses the XML response into a ``TaskCreationPlan``, and optionally
    converts the plan into executable ``CortexTask`` objects.
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
        """Initialize the task creator with a dedicated agent.

        Args:
            verbose (bool): Whether to log creation activity.
                IN: Controls console and logger output.
                OUT: Passed to the internal ``CortexAgent``.
            model (str | None): Model identifier for the creator agent.
                IN: Overrides the default model selection.
                OUT: Set as the ``model`` field of the creator agent.
            llm (BaseLLM | None): LLM backend for the creator agent.
                IN: Passed to the creator agent constructor.
                OUT: Configures the agent's language model.
            max_tasks (int): Maximum number of tasks to allow in a plan.
                IN: Limits the size of parsed plans.
                OUT: Plans with more tasks are truncated.
            auto_assign_agents (bool): Whether to auto-assign agents to tasks.
                IN: If ``True``, ``create_tasks_from_prompt`` returns ``CortexTask``
                objects mapped to available agents.
                OUT: Controls the return type of task creation.
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
        """Generate a task plan from a natural language objective.

        Args:
            prompt (str): The high-level objective to decompose.
                IN: Rendered into the task creation template.
                OUT: Drives the agent's task breakdown.
            background (str | None): Additional context for decomposition.
                IN: Passed to the template to guide the agent's strategy.
                OUT: Rendered conditionally in the prompt.
            available_agents (list | None): Agents that may be assigned to tasks.
                IN: Their roles and goals are rendered into the prompt.
                OUT: Used for agent mapping when ``auto_assign_agents`` is enabled.
            constraints (str | None): Constraints to respect during creation.
                IN: Rendered into the prompt.
                OUT: Guides the agent to respect limitations.
            stream (bool): Whether to stream the agent's response.
                IN: Passed to the creator agent's ``execute`` method.
                OUT: Enables real-time observation of task creation.
            stream_callback (Callable | None): Callback for streamed chunks.
                IN: Invoked during streaming if *stream* is ``True``.
                OUT: Forwarded to the creator agent.
            streamer_buffer (StreamerBuffer | None): Buffer for streaming.
                IN: Forwarded to the creator agent.
                OUT: Receives chunks when streaming is enabled.

        Returns:
            tuple[TaskCreationPlan, list[CortexTask] | None]: The parsed plan and,
                when agents are available and ``auto_assign_agents`` is ``True``,
                a list of executable ``CortexTask`` instances.
                OUT: ``None`` for the task list when auto-assignment is disabled
                or no agents are provided.
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
                raw_response = self.creator_agent.execute(
                    task_description=creation_prompt,
                    streamer_buffer=streamer_buffer,
                    stream_callback=stream_callback,
                )
            else:
                raw_response = self.creator_agent.execute(task_description=creation_prompt)

            if isinstance(raw_response, tuple):
                response = (
                    raw_response[0].get_result(1.0) if raw_response[0].get_result is not None else str(raw_response[0])
                )
            else:
                response = raw_response

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
        """Parse an XML task plan response into a ``TaskCreationPlan``.

        Args:
            xml_response (str): Raw XML or text containing a ``<task_plan>`` element.
                IN: Extracted and parsed as XML.
                OUT: Transformed into ``TaskDefinition`` objects.
            objective (str): The original objective (fallback if not in XML).
                IN: Used when the XML lacks an ``<objective>`` tag.
                OUT: Becomes the plan's objective.

        Returns:
            TaskCreationPlan: The parsed plan with task definitions.
                OUT: Truncated to ``self.max_tasks`` if necessary.

        Raises:
            ValueError: If the XML is malformed or cannot be parsed.
        """

        try:
            xml_match = re.search(r"<task_plan>.*?</task_plan>", xml_response, re.DOTALL)
            if xml_match:
                xml_content = xml_match.group(0)
            else:
                xml_content = xml_response

            root = ET.fromstring(xml_content)

            _objective = root.find("objective")
            _approach = root.find("approach")
            _complexity = root.find("complexity")
            _sequential = root.find("sequential")
            from typing import cast

            plan = TaskCreationPlan(
                plan_id=f"plan_{hash(objective) % 10000}",
                objective=(_objective.text if _objective is not None else None) or objective,
                approach=(_approach.text if _approach is not None else None) or "Standard approach",
                estimated_complexity=cast(
                    Literal["simple", "medium", "complex"],
                    (_complexity.text if _complexity is not None else None) or "medium",
                ),
                sequential=(_sequential.text if _sequential is not None else None) == "true"
                if _sequential is not None
                else True,
            )

            for task_elem in root.findall("task"):
                task_id = int(task_elem.get("id", "0"))

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

                _desc = task_elem.find("description")
                _exp = task_elem.find("expected_output")
                _agent_role = task_elem.find("agent_role")
                _ctx_needed = task_elem.find("context_needed")
                _val_req = task_elem.find("validation_required")
                _hum_fb = task_elem.find("human_feedback")
                task_def = TaskDefinition(
                    task_id=task_id,
                    description=(_desc.text if _desc is not None else None) or "",
                    expected_output=(_exp.text if _exp is not None else None) or "",
                    agent_role=(_agent_role.text if _agent_role is not None else None),
                    dependencies=dependencies,
                    context_needed=(
                        (_ctx_needed.text if _ctx_needed is not None else None) == "true"
                        if _ctx_needed is not None
                        else False
                    ),
                    tools_needed=tools_needed,
                    importance=importance,
                    validation_required=(
                        (_val_req.text if _val_req is not None else None) == "true" if _val_req is not None else False
                    ),
                    human_feedback=(
                        (_hum_fb.text if _hum_fb is not None else None) == "true" if _hum_fb is not None else False
                    ),
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
        """Convert a ``TaskCreationPlan`` into executable ``CortexTask`` objects.

        Args:
            task_plan (TaskCreationPlan): The plan to convert.
                IN: Iterated to create one ``CortexTask`` per ``TaskDefinition``.
                OUT: Task descriptions, expected outputs, and assignments are mapped.
            available_agents (list[CortexAgent]): Agents to assign to tasks.
                IN: Matched by ``agent_role``; falls back to the first agent.
                OUT: Each created task gets an ``agent`` reference.

        Returns:
            list[CortexTask]: Executable tasks in plan order.
                OUT: Dependencies are wired between tasks by index.
        """

        cortex_tasks: list[CortexTask] = []
        agent_map = {agent.role: agent for agent in available_agents}

        for task_def in task_plan.tasks:
            agent = None
            if task_def.agent_role and task_def.agent_role in agent_map:
                agent = agent_map[task_def.agent_role]
            elif available_agents:
                agent = available_agents[0]

            dependencies = [
                cortex_tasks[dep_id - 1]
                for dep_id in task_def.dependencies
                if dep_id > 0 and dep_id - 1 < len(cortex_tasks)
            ]

            cortex_task = CortexTask(
                description=task_def.description,
                expected_output=task_def.expected_output,
                agent=agent,
                importance=task_def.importance,
                human_feedback=task_def.human_feedback,
                context=dependencies if dependencies else None,
                dependencies=dependencies,
            )

            cortex_tasks.append(cortex_task)

        return cortex_tasks

    def _create_fallback_plan(self, objective: str, background: str | None) -> tuple[TaskCreationPlan, None]:
        """Create a minimal fallback plan when decomposition fails.

        Args:
            objective (str): The goal to fall back on.
                IN: Becomes the fallback plan's objective.
                OUT: Stored in the generated plan.
            background (str | None): Additional context for the fallback.
                IN: Used as the plan's approach if provided.
                OUT: Stored in the fallback plan.

        Returns:
            tuple[TaskCreationPlan, None]: A single-task fallback plan with no
                executable ``CortexTask`` list.
                OUT: Ensures the system can still attempt execution.
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
        """Log a summary of the task creation plan.

        Args:
            plan (TaskCreationPlan): The plan to summarize.
                IN: Provides objective, approach, complexity, and task list.
                OUT: Iterated to produce structured log lines.
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
        process_type: ProcessType | None = None,
        use_streaming: bool = False,
        stream_callback: Callable[[Any], None] | None = None,
        log_process: bool = False,
    ) -> Any | tuple[StreamerBuffer, Thread]:
        """Create tasks from a prompt and immediately execute them through a ``Cortex``.

        Args:
            prompt (str): The objective to decompose and execute.
                IN: Passed to ``create_tasks_from_prompt``.
                OUT: Drives both task creation and execution.
            background (str | None): Background context for task creation.
                IN: Passed to ``create_tasks_from_prompt``.
                OUT: Guides the decomposition strategy.
            cortex (Cortex): The orchestration instance to execute through.
                IN: Must have at least one agent defined.
                OUT: Its ``tasks`` list is replaced with the created tasks.
            process_type (ProcessType | None): Optional process type override.
                IN: Temporarily sets ``cortex.process`` during execution.
                OUT: Restored to its original value afterward.
            use_streaming (bool): Whether to stream execution output.
                IN: Passed to ``cortex.kickoff``.
                OUT: Determines the return type.
            stream_callback (Callable | None): Callback for streamed chunks.
                IN: Passed to ``cortex.kickoff``.
                OUT: Invoked during streaming execution.
            log_process (bool): Whether to log the execution process.
                IN: Passed to ``cortex.kickoff``.
                OUT: Controls process-level logging.

        Returns:
            Any | tuple[StreamerBuffer, Thread]: The execution result, or a
                streaming tuple when *use_streaming* is ``True``.
                OUT: Direct return from ``cortex.kickoff``.

        Raises:
            ValueError: If the cortex has no agents defined.
            RuntimeError: If task creation fails to produce executable tasks.
        """

        if cortex.agents:
            _task_plan, cortex_tasks = self.create_tasks_from_prompt(
                prompt=prompt,
                background=background,
                available_agents=cortex.agents,
            )
        else:
            raise ValueError("Cortex must have agents defined")

        if cortex_tasks is None:
            raise RuntimeError("Task creation failed")
        cortex.tasks = cortex_tasks

        if process_type:
            original_process = cortex.process
            cortex.process = process_type

        if use_streaming:
            kickoff_result = cortex.kickoff(use_streaming=True, stream_callback=stream_callback, log_process=log_process)
            assert isinstance(kickoff_result, tuple)
            buffer, thread = kickoff_result
        else:
            result = cortex.kickoff(use_streaming=False, stream_callback=stream_callback, log_process=log_process)

        if process_type:
            cortex.process = original_process
        if use_streaming:
            return buffer, thread
        return result
