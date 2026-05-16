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


"""XML-based planner agent for task orchestration.

This module provides an XML-based planning system for the Cortex framework,
inspired by SmolAgent's planning approach. It enables intelligent task
decomposition by having an LLM generate structured execution plans in XML
format, which can then be executed step by step.

Key features:
- Automatic plan generation from natural language objectives
- XML-based plan format for structured, parseable output
- Dependency tracking between plan steps
- Support for parallel step execution when dependencies allow
- Fallback planning when XML parsing fails
- Streaming support for real-time plan creation feedback

The module provides three main classes:
- PlanStep: Represents a single step in an execution plan
- ExecutionPlan: Complete plan with steps and metadata
- CortexPlanner: The planner agent that creates and executes plans

Typical usage example:
    planner = CortexPlanner(cortex_instance=cortex, verbose=True)


    plan = planner.create_plan(
        objective="Research AI trends and write a summary report",
        available_agents=[researcher, writer]
    )


    results = planner.execute_plan(plan)
"""

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

from ...core.streamer_buffer import StreamerBuffer
from ...logging.console import get_logger
from ...types import StreamChunk
from ..agents.agent import CortexAgent
from ..core.templates import PromptTemplate

if TYPE_CHECKING:
    from .cortex import Cortex
    from .task import CortexTask


@dataclass
class PlanStep:
    """A single step within an execution plan.

    Attributes:
        step_id (int): Unique identifier for the step.
        agent (str): The role name of the agent assigned to this step.
        action (str): The action to perform in this step.
        arguments (dict): Key-value arguments for the action.
        dependencies (list[int]): IDs of steps that must complete before this one.
        description (str): Human-readable description of the step.
    """

    step_id: int
    agent: str
    action: str
    arguments: dict = field(default_factory=dict)
    dependencies: list[int] = field(default_factory=list)
    description: str = ""

    def __str__(self) -> str:
        """Return ``"Step {id}: {agent} -> {action}"`` for log/console output."""

        return f"Step {self.step_id}: {self.agent} -> {self.action}"


@dataclass
class ExecutionPlan:
    """A structured plan composed of ordered steps to achieve an objective.

    Attributes:
        plan_id (str): Unique identifier for the plan.
        objective (str): The goal the plan addresses.
        steps (list[PlanStep]): Ordered list of plan steps.
        estimated_time (float): Estimated execution time in minutes.
        complexity (Literal["low", "medium", "high"]): Estimated complexity.
    """

    plan_id: str
    objective: str
    steps: list[PlanStep] = field(default_factory=list)
    estimated_time: float = 0.0
    complexity: Literal["low", "medium", "high"] = "medium"

    def add_step(self, step: PlanStep):
        """Append ``step`` to :attr:`steps`."""

        self.steps.append(step)

    def get_step(self, step_id: int) -> PlanStep | None:
        """Return the step whose ``step_id`` matches, or ``None`` if absent."""

        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_next_steps(self, completed_steps: set[int]) -> list[PlanStep]:
        """Return all not-yet-run steps whose dependencies are all satisfied."""

        next_steps = []
        for step in self.steps:
            if step.step_id not in completed_steps:
                if all(dep_id in completed_steps for dep_id in step.dependencies):
                    next_steps.append(step)
        return next_steps


class CortexPlanner:
    """Generates and executes structured execution plans for complex objectives.

    Uses a dedicated ``CortexAgent`` as a strategic planner to create XML-based
    execution plans, then executes them step-by-step against a ``Cortex`` instance.
    """

    def __init__(
        self,
        cortex_instance: Optional["Cortex"] = None,
        verbose: bool = True,
        planning_model: str | None = None,
    ):
        """Build a dedicated planner :class:`CortexAgent` and keep a Cortex ref.

        The planner agent itself runs without delegation; ``cortex_instance``
        is captured so :meth:`execute_plan` can dispatch step work to the
        regular agents.
        """

        self.cortex_instance = cortex_instance
        self.verbose = verbose
        self.planning_model = planning_model
        self.logger = get_logger() if verbose else None
        self.template_engine = PromptTemplate()

        self.planner_agent = CortexAgent(
            role="Strategic Planner",
            goal="Create detailed execution plans for complex objectives",
            backstory="""You are an expert strategic planner who breaks down complex objectives
            into actionable steps. You understand agent capabilities and can create efficient
            execution plans using XML format.""",
            model=planning_model,
            verbose=verbose,
            allow_delegation=False,
        )

    def create_plan(
        self,
        objective: str,
        available_agents: list[CortexAgent],
        context: str = "",
        streamer_buffer: StreamerBuffer | None = None,
        stream_callback: Callable[[Any], None] | None = None,
    ) -> ExecutionPlan:
        """Ask the planner agent for an XML plan and parse it.

        On parse failure, a minimal one-step fallback plan is returned
        instead of raising so execution can still proceed.
        """

        if self.verbose and self.logger:
            self.logger.info(f"🧠 Planner creating plan for: {objective[:100]}...")

        planning_prompt = self.template_engine.render_planner(
            objective=objective,
            agents=available_agents,
            context=context,
        )

        try:
            if not self.planner_agent.xerxes_instance and self.cortex_instance:
                self.planner_agent.xerxes_instance = self.cortex_instance.xerxes

            raw_response = self.planner_agent.execute(
                task_description=planning_prompt,
                context=context,
                streamer_buffer=streamer_buffer,
                stream_callback=stream_callback,
            )
            if isinstance(raw_response, tuple):
                plan_response = (
                    raw_response[0].get_result(1.0) if raw_response[0].get_result is not None else str(raw_response[0])
                )
            else:
                plan_response = raw_response

            execution_plan = self._parse_xml_plan(plan_response, objective)

            if self.verbose:
                success_msg = f"✅ Plan created with {len(execution_plan.steps)} steps"
                if stream_callback:
                    stream_callback(success_msg)
                if streamer_buffer:
                    streamer_buffer.put(
                        StreamChunk(
                            chunk=None,
                            agent_id="planner",
                            content=success_msg + "\n",
                            buffered_content=success_msg + "\n",
                            function_calls_detected=False,
                            reinvoked=False,
                        )
                    )
                if self.logger:
                    self.logger.info(success_msg)
                self._log_plan_summary(execution_plan)

            return execution_plan

        except Exception as e:
            error_msg = f"❌ Failed to create plan: {e}"
            if stream_callback:
                stream_callback(error_msg)
            if streamer_buffer:
                streamer_buffer.put(
                    StreamChunk(
                        chunk=None,
                        agent_id="planner",
                        content=error_msg + "\n",
                        buffered_content=error_msg + "\n",
                        function_calls_detected=False,
                        reinvoked=False,
                    )
                )
            if self.verbose and self.logger:
                self.logger.error(error_msg)

            return self._create_fallback_plan(objective, available_agents)

    def execute_plan(self, plan: ExecutionPlan, tasks: list["CortexTask"] | None = None) -> dict:
        """Walk the plan, dispatching ready steps until everything is done.

        Each iteration calls :meth:`ExecutionPlan.get_next_steps`; if none
        are eligible the loop exits to avoid spinning on a circular
        dependency. Errors per step are logged and the step is marked
        completed so the rest of the plan can still proceed.

        Returns:
            Mapping from ``step_id`` to the step's output string.

        Raises:
            ValueError: When the planner has no Cortex bound.
        """

        if not self.cortex_instance:
            raise ValueError("Cortex instance required for plan execution")

        if self.verbose:
            if self.logger:
                self.logger.info(f"🚀 Executing plan: {plan.objective}")

        task_context = ""
        if tasks:
            task_context = "Original tasks context:\n"
            for i, task in enumerate(tasks, 1):
                task_context += f"{i}. {task.description}\n"
                if task.expected_output:
                    task_context += f"   Expected: {task.expected_output}\n"
            task_context += "\n"

        completed_steps: set[int] = set()
        step_results: dict[int, str] = {}

        while len(completed_steps) < len(plan.steps):
            next_steps = plan.get_next_steps(completed_steps)

            if not next_steps:
                if self.logger:
                    self.logger.error("❌ No executable steps found - possible circular dependency")
                break

            for step in next_steps:
                try:
                    result = self._execute_step(step, step_results, task_context)
                    step_results[step.step_id] = result
                    completed_steps.add(step.step_id)

                    if self.verbose:
                        if self.logger:
                            self.logger.info(f"✅ Step {step.step_id} completed")

                except Exception as e:
                    if self.verbose:
                        if self.logger:
                            self.logger.error(f"❌ Step {step.step_id} failed: {e}")

                    completed_steps.add(step.step_id)

        if self.verbose:
            if self.logger:
                self.logger.info("🎉 Plan execution completed")

        return step_results

    def _format_agents_info(self, agents: list[CortexAgent]) -> str:
        """Render ``agents`` as bulleted ``- role: goal (Tools: ...)`` lines."""

        agents_info = []
        for agent in agents:
            info = f"- {agent.role}: {agent.goal}"
            if agent.tools:
                tools = ", ".join([tool.__class__.__name__ for tool in agent.tools])
                info += f" (Tools: {tools})"
            agents_info.append(info)
        return "\n".join(agents_info)

    def _build_planning_prompt(self, objective: str, agents_info: str, context: str) -> str:
        """Build the verbatim XML-plan prompt as a fallback to the template engine."""

        return f"""
You are a strategic planner. Create a detailed execution plan for the following objective.

OBJECTIVE: {objective}

AVAILABLE AGENTS:
{agents_info}

CONTEXT: {context or "No additional context provided"}

Create a plan using the following XML format:

<plan>
    <objective>{objective}</objective>
    <complexity>low|medium|high</complexity>
    <estimated_time>minutes</estimated_time>

    <step id="1">
        <agent>Agent Role Name</agent>
        <action>specific_action_to_take</action>
        <arguments>
            <key1>value1</key1>
            <key2>value2</key2>
        </arguments>
        <dependencies></dependencies>
        <description>Clear description of what this step accomplishes</description>
    </step>

    <step id="2">
        <agent>Another Agent Role Name</agent>
        <action>another_action</action>
        <arguments>
            <input>result_from_step_1</input>
        </arguments>
        <dependencies>1</dependencies>
        <description>This step depends on step 1 completion</description>
    </step>
</plan>

INSTRUCTIONS:
1. Break down the objective into logical, sequential steps
2. Assign each step to the most appropriate agent based on their role and capabilities
3. Specify clear dependencies between steps (use step IDs)
4. Include all necessary arguments for each action
5. Make sure the plan is executable and complete
6. Use specific action names like: research, write, analyze, review, create, etc.

Respond ONLY with the XML plan, no additional text.
"""

    def _parse_xml_plan(self, xml_response: str, objective: str) -> ExecutionPlan:
        """Extract and parse the ``<plan>`` block into an :class:`ExecutionPlan`.

        Raises:
            ValueError: When the response contains no parseable plan XML.
        """

        try:
            xml_match = re.search(r"<plan>.*?</plan>", xml_response, re.DOTALL)
            if xml_match:
                xml_content = xml_match.group(0)
            else:
                xml_content = xml_response

            root = ET.fromstring(xml_content)

            _objective = root.find("objective")
            _complexity = root.find("complexity")
            _estimated_time = root.find("estimated_time")

            from typing import cast

            plan = ExecutionPlan(
                plan_id=f"plan_{hash(objective) % 10000}",
                objective=(_objective.text if _objective is not None else None) or objective,
                complexity=cast(
                    Literal["low", "medium", "high"], (_complexity.text if _complexity is not None else None) or "medium"
                ),
                estimated_time=float((_estimated_time.text if _estimated_time is not None else None) or 10),
            )

            for step_elem in root.findall("step"):
                step_id = int(step_elem.get("id", "0"))
                _agent_elem = step_elem.find("agent")
                _action_elem = step_elem.find("action")
                _description_elem = step_elem.find("description")
                agent = (_agent_elem.text if _agent_elem is not None else None) or ""
                action = (_action_elem.text if _action_elem is not None else None) or ""
                description = (_description_elem.text if _description_elem is not None else None) or ""

                arguments = {}
                args_elem = step_elem.find("arguments")
                if args_elem is not None:
                    for arg in args_elem:
                        arguments[arg.tag] = arg.text

                dependencies = []
                deps_elem = step_elem.find("dependencies")
                if deps_elem is not None and deps_elem.text:
                    deps_text = deps_elem.text.strip()
                    if deps_text:
                        dependencies = [int(x.strip()) for x in deps_text.split(",")]

                step = PlanStep(
                    step_id=step_id,
                    agent=agent,
                    action=action,
                    arguments=arguments,
                    dependencies=dependencies,
                    description=description,
                )
                plan.add_step(step)

            return plan

        except Exception as e:
            if self.verbose:
                if self.logger:
                    self.logger.error(f"❌ Failed to parse XML plan: {e}")
            raise ValueError(f"Invalid XML plan format: {e}") from e

    def _create_fallback_plan(self, objective: str, agents: list[CortexAgent]) -> ExecutionPlan:
        """Return a single-step plan assigned to ``agents[0]`` for ``objective``."""

        plan = ExecutionPlan(
            plan_id=f"fallback_{hash(objective) % 10000}", objective=objective, complexity="low", estimated_time=5.0
        )

        if agents:
            step = PlanStep(
                step_id=1,
                agent=agents[0].role,
                action="execute_objective",
                arguments={"objective": objective},
                description=f"Execute objective using {agents[0].role}",
            )
            plan.add_step(step)

        return plan

    def _execute_step(self, step: PlanStep, previous_results: dict, task_context: str = "") -> str:
        """Resolve ``step``'s agent and arguments, then run it through Cortex.

        Argument values of the form ``result_from_step_N`` are replaced
        with the cached output of step ``N`` before the agent is invoked.

        Raises:
            ValueError: When the assigned agent role is not registered with
                the bound Cortex.
        """

        if not self.cortex_instance:
            raise ValueError("Cortex instance required")

        agent = None
        for a in self.cortex_instance.agents:
            if a.role.lower() == step.agent.lower():
                agent = a
                break

        if not agent:
            raise ValueError(f"Agent '{step.agent}' not found")

        task_description = f"Action: {step.action}\n"
        task_description += f"Description: {step.description}\n"

        if step.arguments:
            task_description += "Arguments:\n"
            for key, value in step.arguments.items():
                if isinstance(value, str) and value.startswith("result_from_step_"):
                    step_ref = int(value.split("_")[-1])
                    if step_ref in previous_results:
                        value = previous_results[step_ref]
                task_description += f"- {key}: {value}\n"

        context_parts = []

        if task_context:
            context_parts.append(task_context.strip())

        if step.dependencies:
            for dep_id in step.dependencies:
                if dep_id in previous_results:
                    context_parts.append(f"Result from step {dep_id}: {previous_results[dep_id]}")

        context = "\n\n".join(context_parts) if context_parts else ""

        if self.verbose:
            if self.logger:
                self.logger.info(f"🔄 Executing step {step.step_id}: {step.agent} -> {step.action}")

        if agent.allow_delegation:
            delegated = agent.execute_with_delegation(task_description, context)
            return delegated
        executed = agent.execute(task_description, context)
        if isinstance(executed, tuple):
            return executed[0].get_result(1.0) if executed[0].get_result is not None else str(executed[0])
        return executed

    def _log_plan_summary(self, plan: ExecutionPlan):
        """Emit a multi-line summary of ``plan`` to the logger."""

        if self.logger:
            self.logger.info("📋 Plan Summary:")
            self.logger.info(f"  • Objective: {plan.objective}")
            self.logger.info(f"  • Steps: {len(plan.steps)}")
            self.logger.info(f"  • Complexity: {plan.complexity}")
            self.logger.info(f"  • Estimated time: {plan.estimated_time} minutes")

        for step in plan.steps:
            deps = f" (depends on: {step.dependencies})" if step.dependencies else ""
            if self.logger:
                self.logger.info(f"    {step.step_id}. {step.agent} -> {step.action}{deps}")
