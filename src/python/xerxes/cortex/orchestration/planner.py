# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

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
from ..agents.agent import CortexAgent
from ..core.templates import PromptTemplate

if TYPE_CHECKING:
    from .cortex import Cortex
    from .task import CortexTask


@dataclass
class PlanStep:
    """A single step in the execution plan.

    PlanStep represents an atomic unit of work within an ExecutionPlan.
    Each step is assigned to a specific agent and defines an action to
    perform along with any required arguments and dependencies on other steps.

    Attributes:
        step_id: Unique integer identifier for this step within the plan.
        agent: The role name of the agent assigned to execute this step.
        action: The action verb describing what the step does (e.g., "research",
            "write", "analyze").
        arguments: Dictionary of key-value pairs providing input parameters
            for the action. May include references to results from previous steps.
        dependencies: List of step IDs that must complete before this step
            can execute. Empty list means the step can run immediately.
        description: Human-readable description of what this step accomplishes.

    Example:
        step = PlanStep(
            step_id=2,
            agent="Writer",
            action="write_draft",
            arguments={"topic": "AI trends", "input": "result_from_step_1"},
            dependencies=[1],
            description="Write initial draft based on research findings"
        )
    """

    step_id: int
    agent: str
    action: str
    arguments: dict = field(default_factory=dict)
    dependencies: list[int] = field(default_factory=list)
    description: str = ""

    def __str__(self) -> str:
        """Return string representation of the step.

        Returns:
            str: Formatted string showing step ID, agent, and action.
        """
        return f"Step {self.step_id}: {self.agent} -> {self.action}"


@dataclass
class ExecutionPlan:
    """Complete execution plan with steps and metadata.

    ExecutionPlan represents a complete plan for achieving an objective,
    consisting of multiple PlanStep instances with their dependencies.
    It provides methods for managing steps and determining execution order
    based on dependency resolution.

    The plan tracks metadata like complexity and estimated time to help
    with resource allocation and progress estimation.

    Attributes:
        plan_id: Unique identifier for this plan, typically derived from
            the objective hash.
        objective: The high-level goal this plan is designed to achieve.
        steps: List of PlanStep instances that make up the plan.
        estimated_time: Estimated execution time in minutes. Used for
            planning and progress tracking.
        complexity: Plan complexity rating ("low", "medium", or "high").
            Affects resource allocation and timeout settings.

    Example:
        plan = ExecutionPlan(
            plan_id="plan_1234",
            objective="Create a comprehensive market analysis",
            complexity="high",
            estimated_time=30.0
        )
        plan.add_step(research_step)
        plan.add_step(analysis_step)


        ready_steps = plan.get_next_steps(completed_steps=set())
    """

    plan_id: str
    objective: str
    steps: list[PlanStep] = field(default_factory=list)
    estimated_time: float = 0.0
    complexity: Literal["low", "medium", "high"] = "medium"

    def add_step(self, step: PlanStep):
        """Add a step to the plan.

        Appends a new PlanStep to the end of the steps list.

        Args:
            step: The PlanStep instance to add to the plan.
        """
        self.steps.append(step)

    def get_step(self, step_id: int) -> PlanStep | None:
        """Get step by ID.

        Searches through the plan's steps to find one with the matching ID.

        Args:
            step_id: The unique identifier of the step to retrieve.

        Returns:
            PlanStep: The step with the matching ID, or None if not found.
        """
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_next_steps(self, completed_steps: set[int]) -> list[PlanStep]:
        """Get steps that can be executed next based on dependencies.

        Analyzes the plan to find all steps whose dependencies have been
        satisfied (all dependent steps are in the completed set). This
        enables parallel execution of independent steps.

        Args:
            completed_steps: Set of step IDs that have already completed
                execution successfully.

        Returns:
            list[PlanStep]: List of steps that are ready to execute. These
                steps have not yet been completed and have all their
                dependencies satisfied. May return multiple steps if they
                can be executed in parallel.
        """
        next_steps = []
        for step in self.steps:
            if step.step_id not in completed_steps:
                if all(dep_id in completed_steps for dep_id in step.dependencies):
                    next_steps.append(step)
        return next_steps


class CortexPlanner:
    """XML-based planner agent for task orchestration.

    CortexPlanner uses an LLM-powered agent to analyze objectives and create
    structured execution plans in XML format. The planner understands agent
    capabilities and creates efficient plans with proper dependency ordering.

    The planner operates in two phases:
    1. Plan Creation: Generates an ExecutionPlan from a natural language objective
    2. Plan Execution: Executes the plan step by step, respecting dependencies

    Attributes:
        cortex_instance: Reference to the parent Cortex instance for access
            to agents and execution context.
        verbose: Whether to output detailed logging information.
        planning_model: Optional model identifier for the planner agent.
        logger: Logger instance for verbose output (None if verbose=False).
        template_engine: PromptTemplate instance for generating planning prompts.
        planner_agent: CortexAgent configured as a strategic planner.

    Example:
        planner = CortexPlanner(
            cortex_instance=cortex,
            verbose=True,
            planning_model="gpt-4"
        )

        plan = planner.create_plan(
            objective="Build a data pipeline",
            available_agents=[analyst, engineer]
        )

        results = planner.execute_plan(plan)
    """

    def __init__(
        self,
        cortex_instance: Optional["Cortex"] = None,
        verbose: bool = True,
        planning_model: str | None = None,
    ):
        """Initialize the CortexPlanner with configuration.

        Creates a new planner instance with an internal planner agent
        configured for strategic planning tasks.

        Args:
            cortex_instance: Optional reference to the parent Cortex instance.
                Required for plan execution but optional for plan creation.
            verbose: Whether to enable detailed logging output. Defaults to True.
            planning_model: Optional model identifier for the planner agent
                (e.g., "gpt-4"). If None, uses the agent's default.
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
        """Create an execution plan for the given objective.

        Uses the planner agent to analyze the objective and available agents,
        then generates a structured ExecutionPlan with properly ordered steps.
        Supports streaming for real-time feedback during plan creation.

        Args:
            objective: The high-level goal to create a plan for.
            available_agents: List of CortexAgent instances that can be
                assigned to plan steps.
            context: Optional additional context to guide plan creation.
            streamer_buffer: Optional StreamerBuffer for streaming output.
            stream_callback: Optional callback function for streaming chunks.

        Returns:
            ExecutionPlan: A complete plan with steps, dependencies, and
                metadata. Returns a fallback plan if XML parsing fails.

        Raises:
            No exceptions are raised; failures result in fallback plans.
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

            plan_response = self.planner_agent.execute(
                task_description=planning_prompt,
                context=context,
                streamer_buffer=streamer_buffer,
                stream_callback=stream_callback,
            )

            execution_plan = self._parse_xml_plan(plan_response, objective)

            if self.verbose:
                success_msg = f"✅ Plan created with {len(execution_plan.steps)} steps"
                if stream_callback:
                    stream_callback(success_msg)
                if streamer_buffer:
                    streamer_buffer.put(success_msg + "\n")
                if self.logger:
                    self.logger.info(success_msg)
                self._log_plan_summary(execution_plan)

            return execution_plan

        except Exception as e:
            error_msg = f"❌ Failed to create plan: {e}"
            if stream_callback:
                stream_callback(error_msg)
            if streamer_buffer:
                streamer_buffer.put(error_msg + "\n")
            if self.verbose and self.logger:
                self.logger.error(error_msg)

            return self._create_fallback_plan(objective, available_agents)

    def execute_plan(self, plan: ExecutionPlan, tasks: list["CortexTask"] | None = None) -> dict:
        """Execute the plan step by step.

        Iterates through plan steps, respecting dependencies, and executes
        each step using the assigned agent. Results from previous steps are
        passed as context to dependent steps.

        Args:
            plan: The ExecutionPlan to execute.
            tasks: Optional list of original CortexTask objects to provide
                additional context during execution.

        Returns:
            dict: Dictionary mapping step IDs to their execution results.
                Each key is a step_id (int) and value is the result string.

        Raises:
            ValueError: If cortex_instance is not set (required for execution).
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

        completed_steps = set()
        step_results = {}

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
        """Format agent information for planning prompt.

        Creates a formatted string describing all available agents and
        their capabilities for inclusion in the planning prompt.

        Args:
            agents: List of CortexAgent instances to format.

        Returns:
            str: Newline-separated string with agent roles, goals, and tools.
        """
        agents_info = []
        for agent in agents:
            info = f"- {agent.role}: {agent.goal}"
            if agent.tools:
                tools = ", ".join([tool.__class__.__name__ for tool in agent.tools])
                info += f" (Tools: {tools})"
            agents_info.append(info)
        return "\n".join(agents_info)

    def _build_planning_prompt(self, objective: str, agents_info: str, context: str) -> str:
        """Build the planning prompt with XML format requirements.

        Constructs a detailed prompt that instructs the planner agent to
        create an execution plan in the expected XML format.

        Args:
            objective: The goal to plan for.
            agents_info: Formatted string describing available agents.
            context: Additional context to include in the prompt.

        Returns:
            str: Complete planning prompt with XML template and instructions.
        """
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
        """Parse XML plan response into ExecutionPlan object.

        Extracts the XML plan from the LLM response and converts it into
        an ExecutionPlan with PlanStep instances. Handles extraction of
        the plan from mixed text/XML responses.

        Args:
            xml_response: The raw response from the planner agent,
                potentially containing XML within other text.
            objective: The original objective, used as fallback if not
                found in the XML.

        Returns:
            ExecutionPlan: Parsed plan with all steps and metadata.

        Raises:
            ValueError: If the XML is malformed or cannot be parsed.
        """
        try:
            xml_match = re.search(r"<plan>.*?</plan>", xml_response, re.DOTALL)
            if xml_match:
                xml_content = xml_match.group(0)
            else:
                xml_content = xml_response

            root = ET.fromstring(xml_content)

            plan = ExecutionPlan(
                plan_id=f"plan_{hash(objective) % 10000}",
                objective=root.find("objective").text or objective,
                complexity=root.find("complexity").text or "medium",
                estimated_time=float(root.find("estimated_time").text or 10),
            )

            for step_elem in root.findall("step"):
                step_id = int(step_elem.get("id"))
                agent = step_elem.find("agent").text
                action = step_elem.find("action").text
                description = step_elem.find("description").text or ""

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
        """Create a simple fallback plan if XML parsing fails.

        Generates a minimal plan with a single step when the normal
        planning process fails. Uses the first available agent to
        execute the objective directly.

        Args:
            objective: The original objective that couldn't be planned.
            agents: List of available agents; first one will be used.

        Returns:
            ExecutionPlan: Simple plan with one step using the first agent.
        """
        plan = ExecutionPlan(
            plan_id=f"fallback_{hash(objective) % 10000}", objective=objective, complexity="simple", estimated_time=5.0
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
        """Execute a single plan step.

        Finds the appropriate agent and executes the step's action with
        the specified arguments. Resolves references to previous step
        results in the arguments.

        Args:
            step: The PlanStep to execute.
            previous_results: Dictionary mapping step IDs to their results,
                used to resolve references like "result_from_step_1".
            task_context: Optional context string from original tasks.

        Returns:
            str: The result of executing the step.

        Raises:
            ValueError: If cortex_instance is not set or agent not found.
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
            result = agent.execute_with_delegation(task_description, context)
        else:
            result = agent.execute(task_description, context)

        return result

    def _log_plan_summary(self, plan: ExecutionPlan):
        """Log a summary of the execution plan.

        Outputs a formatted summary of the plan including its objective,
        step count, complexity, estimated time, and each step's details.

        Args:
            plan: The ExecutionPlan to summarize.

        Note:
            Only logs if a logger is available (verbose mode enabled).
        """
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
