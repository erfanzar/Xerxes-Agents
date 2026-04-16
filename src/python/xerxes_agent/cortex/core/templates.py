# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
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


"""Jinja2 template support for Cortex prompts.

This module provides a comprehensive template system for generating prompts
used throughout the Cortex framework. It includes:
- Jinja2-based template rendering with automatic fallback
- Pre-defined templates for agents, tasks, delegation, and planning
- Template variable extraction and validation
- Custom template creation support

The template system is designed to generate consistent, well-structured
prompts for LLM interactions across different orchestration scenarios
including sequential, hierarchical, parallel, and consensus-based workflows.

Example:
    >>> template = PromptTemplate()
    >>> prompt = template.render_agent_prompt(
    ...     role="Data Analyst",
    ...     goal="Analyze sales data",
    ...     backstory="Expert in statistical analysis"
    ... )
    >>> task_prompt = template.render_task_prompt(
    ...     description="Analyze Q4 sales trends",
    ...     expected_output="A detailed report with insights"
    ... )
"""

from jinja2 import Environment, Template, meta


class PromptTemplate:
    """Prompt template engine with Jinja2 support for Cortex framework.

    PromptTemplate provides a unified interface for rendering prompts used
    in AI agent interactions. It supports both Jinja2 templates (with
    conditional logic, loops, etc.) and simple string formatting as a fallback.

    The class includes pre-defined templates for common orchestration patterns:
    - Agent system prompts with role, goal, and backstory
    - Task prompts with context and constraints
    - Manager delegation and review prompts
    - Multi-agent consensus synthesis prompts
    - Strategic planning prompts for workflow generation

    Attributes:
        use_jinja: Whether to use Jinja2 for template rendering. Defaults to True.
        env: Jinja2 Environment instance configured for prompt generation.

    Class Attributes:
        AGENT_TEMPLATE: Template for generating agent system prompts.
        TASK_TEMPLATE: Template for generating task execution prompts.
        MANAGER_DELEGATION_TEMPLATE: Template for manager task assignment.
        MANAGER_REVIEW_TEMPLATE: Template for reviewing agent outputs.
        CONSENSUS_TEMPLATE: Template for synthesizing multi-agent outputs.
        PLANNER_TEMPLATE: Template for generating XML execution plans.
        STEP_EXECUTION_TEMPLATE: Template for executing individual plan steps.

    Example:
        >>> template = PromptTemplate()
        >>> agent_prompt = template.render_agent_prompt(
        ...     role="Researcher",
        ...     goal="Find relevant papers",
        ...     backstory="PhD in Computer Science",
        ...     tools=[{"name": "search", "description": "Search papers"}]
        ... )
    """

    AGENT_TEMPLATE = """
You are {{ role }}.
Goal: {{ goal }}
Backstory: {{ backstory }}

{% if instructions %}
Instructions:
{{ instructions }}
{% endif %}

{% if rules %}
Rules:
{% for rule in rules %}
- {{ rule }}
{% endfor %}
{% endif %}

{% if tools %}
Available Tools:
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{% endfor %}
{% endif %}

You must work towards achieving your goal while following your role's responsibilities.
When using tools, always provide clear and detailed responses.
"""

    TASK_TEMPLATE = """
{% if context %}
Context from previous tasks:
{{ context }}

{% endif %}
Task: {{ description }}

Expected Output: {{ expected_output }}

{% if constraints %}
Constraints:
{% for constraint in constraints %}
- {{ constraint }}
{% endfor %}
{% endif %}

Please complete this task according to your role and capabilities.
"""

    MANAGER_DELEGATION_TEMPLATE = """
You are managing a team with the following agents:
{% for agent in agents %}
- {{ agent.role }}: {{ agent.goal }}
{% endfor %}

Tasks to complete:
{% for task in tasks %}
{{ loop.index }}. {{ task.description }}
   Expected: {{ task.expected_output }}
{% endfor %}

Create an execution plan that:
1. Assigns each task to the most appropriate agent based on their expertise
2. Defines the order of execution considering dependencies
3. Identifies potential bottlenecks or challenges
4. Suggests optimizations for efficiency

Return your plan in the following JSON format:
{
  "execution_plan": [
    {
      "task_id": 1,
      "assigned_to": "agent_role",
      "reason": "why this agent is best suited",
      "dependencies": [],
      "estimated_complexity": "low|medium|high"
    }
  ],
  "optimizations": ["suggestion1", "suggestion2"],
  "risks": ["risk1", "risk2"]
}
"""

    MANAGER_REVIEW_TEMPLATE = """
Review the following output from {{ agent_role }}:

Task: {{ task_description }}
Output: {{ output }}

Evaluate the output based on:
1. Completeness - Does it fully address the task?
2. Quality - Is the work of high standard?
3. Accuracy - Are there any errors or inconsistencies?
4. Alignment - Does it meet the expected output requirements?

Provide your assessment in the following format:
{
  "approved": true/false,
  "score": 0-100,
  "feedback": "detailed feedback",
  "improvements_needed": ["improvement1", "improvement2"],
  "strengths": ["strength1", "strength2"]
}
"""

    CONSENSUS_TEMPLATE = """
Multiple agents have provided their perspectives on the following task:
{{ task_description }}

Agent Outputs:
{% for agent_role, output in agent_outputs.items() %}
{{ agent_role }}:
{{ output }}

{% endfor %}

Synthesize these outputs into a unified response that:
1. Incorporates the best insights from all agents
2. Resolves any contradictions or conflicts
3. Provides a comprehensive and balanced perspective
4. Maintains coherence and clarity

Create a consensus response that represents the collective intelligence of the team.
"""

    PLANNER_TEMPLATE = """
You are a strategic planner. Create a detailed execution plan for the following objective.

OBJECTIVE: {{ objective }}

AVAILABLE AGENTS:
{% for agent in agents %}
- {{ agent.role }}: {{ agent.goal }}{% if agent.tools %} (Tools: {% for tool in agent.tools %}{{ tool.__class__.__name__ }}{% if not loop.last %}, {% endif %}{% endfor %}){% endif %}
{% endfor %}

{% if context %}
CONTEXT: {{ context }}
{% else %}
CONTEXT: No additional context provided
{% endif %}

Create a plan using the following XML format:

<plan>
    <objective>{{ objective }}</objective>
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

    STEP_EXECUTION_TEMPLATE = """
You are executing a planned step in a larger workflow.

STEP DETAILS:
- Action: {{ action }}
- Description: {{ description }}

{% if arguments %}
ARGUMENTS:
{% for key, value in arguments.items() %}
- {{ key }}: {{ value }}
{% endfor %}
{% endif %}

{% if context %}
CONTEXT FROM PREVIOUS STEPS:
{{ context }}
{% endif %}

Execute this step thoroughly and provide a clear result that can be used by subsequent steps in the workflow.
"""

    def __init__(self):
        """Initialize the PromptTemplate engine.

        Creates a Jinja2 Environment configured for prompt generation with:
        - trim_blocks: Removes first newline after block tags
        - lstrip_blocks: Strips leading whitespace from block lines
        - keep_trailing_newline: Disabled to avoid extra newlines
        """
        self.use_jinja = True
        self.env = Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False)

    def render(self, template: str, **kwargs) -> str:
        """Render a template with the given context variables.

        Uses Jinja2 templating if enabled, otherwise falls back to Python's
        string format method. Template errors are caught and the fallback
        method is used automatically.

        Args:
            template: The template string containing Jinja2 or format placeholders.
            **kwargs: Context variables to substitute into the template.

        Returns:
            The rendered template string with all variables substituted.

        Example:
            >>> template = PromptTemplate()
            >>> result = template.render("Hello {{ name }}!", name="World")
            >>> print(result)
            Hello World!
        """
        if not self.use_jinja:
            return template.format(**kwargs)

        try:
            tmpl = self.env.from_string(template)
            return tmpl.render(**kwargs)
        except Exception as e:
            print(f"Template rendering error: {e}")

            return template.format(**kwargs)

    def render_agent_prompt(
        self,
        role: str,
        goal: str,
        backstory: str,
        instructions: str | None = None,
        rules: list | None = None,
        tools: list | None = None,
    ) -> str:
        """Render an agent system prompt from role, goal, and backstory.

        Generates a comprehensive system prompt that defines the agent's identity,
        objectives, and capabilities for LLM interactions.

        Args:
            role: The agent's role or title (e.g., "Data Analyst", "Researcher").
            goal: The primary objective the agent should work towards.
            backstory: Background information providing context for the agent's expertise.
            instructions: Optional custom instructions for the agent's behavior.
            rules: Optional list of rules the agent must follow.
            tools: Optional list of tool objects with 'name' and 'description' attributes.

        Returns:
            A formatted system prompt string for the agent.

        Example:
            >>> template = PromptTemplate()
            >>> prompt = template.render_agent_prompt(
            ...     role="Code Reviewer",
            ...     goal="Review code for quality and best practices",
            ...     backstory="Senior engineer with 10 years experience",
            ...     rules=["Be constructive", "Explain your reasoning"]
            ... )
        """
        return self.render(
            self.AGENT_TEMPLATE,
            role=role,
            goal=goal,
            backstory=backstory,
            instructions=instructions,
            rules=rules,
            tools=tools,
        )

    def render_task_prompt(
        self,
        description: str,
        expected_output: str,
        context: str | None = None,
        constraints: list | None = None,
    ) -> str:
        """Render a task execution prompt for an agent.

        Generates a prompt that describes a task to be completed, including
        context from previous tasks and any constraints that must be followed.

        Args:
            description: The task description explaining what needs to be done.
            expected_output: Description of the expected output format and content.
            context: Optional context from previous task outputs to inform this task.
            constraints: Optional list of constraints the agent must adhere to.

        Returns:
            A formatted task prompt string.

        Example:
            >>> template = PromptTemplate()
            >>> prompt = template.render_task_prompt(
            ...     description="Summarize the quarterly report",
            ...     expected_output="A 3-paragraph executive summary",
            ...     constraints=["Use formal language", "Include key metrics"]
            ... )
        """
        return self.render(
            self.TASK_TEMPLATE,
            description=description,
            expected_output=expected_output,
            context=context,
            constraints=constraints,
        )

    def render_manager_delegation(self, agents: list, tasks: list) -> str:
        """Render a manager delegation prompt for hierarchical orchestration.

        Generates a prompt for a manager agent to create an execution plan
        that assigns tasks to the most appropriate team agents based on their
        expertise and capabilities.

        Args:
            agents: List of agent objects with 'role' and 'goal' attributes.
            tasks: List of task objects with 'description' and 'expected_output' attributes.

        Returns:
            A formatted delegation prompt requesting a JSON execution plan.

        Example:
            >>> template = PromptTemplate()
            >>> prompt = template.render_manager_delegation(
            ...     agents=[{"role": "Writer", "goal": "Write content"}],
            ...     tasks=[{"description": "Write article", "expected_output": "1000 words"}]
            ... )
        """
        return self.render(
            self.MANAGER_DELEGATION_TEMPLATE,
            agents=agents,
            tasks=tasks,
        )

    def render_manager_review(
        self,
        agent_role: str,
        task_description: str,
        output: str,
    ) -> str:
        """Render a manager review prompt for evaluating agent outputs.

        Generates a prompt for a manager agent to assess the quality of
        an agent's task output, including completeness, quality, accuracy,
        and alignment with requirements.

        Args:
            agent_role: The role of the agent whose output is being reviewed.
            task_description: The original task description.
            output: The agent's output to be reviewed.

        Returns:
            A formatted review prompt requesting a JSON assessment.

        Example:
            >>> template = PromptTemplate()
            >>> prompt = template.render_manager_review(
            ...     agent_role="Writer",
            ...     task_description="Write an article about AI",
            ...     output="AI is transforming industries..."
            ... )
        """
        return self.render(
            self.MANAGER_REVIEW_TEMPLATE,
            agent_role=agent_role,
            task_description=task_description,
            output=output,
        )

    def render_consensus(
        self,
        task_description: str,
        agent_outputs: dict[str, str],
    ) -> str:
        """Render a consensus synthesis prompt for multi-agent outputs.

        Generates a prompt for synthesizing outputs from multiple agents
        into a unified, coherent response that incorporates the best
        insights from all perspectives.

        Args:
            task_description: The original task that all agents addressed.
            agent_outputs: Dictionary mapping agent roles to their outputs.

        Returns:
            A formatted consensus prompt for synthesizing multiple perspectives.

        Example:
            >>> template = PromptTemplate()
            >>> prompt = template.render_consensus(
            ...     task_description="Evaluate the new product design",
            ...     agent_outputs={
            ...         "Designer": "The aesthetics are excellent...",
            ...         "Engineer": "The feasibility is good..."
            ...     }
            ... )
        """
        return self.render(
            self.CONSENSUS_TEMPLATE,
            task_description=task_description,
            agent_outputs=agent_outputs,
        )

    def render_planner(
        self,
        objective: str,
        agents: list,
        context: str = "",
    ) -> str:
        """Render a strategic planner prompt for XML execution plan generation.

        Generates a prompt for creating a detailed execution plan that breaks
        down an objective into sequential steps, assigns agents, and specifies
        dependencies between steps.

        Args:
            objective: The high-level objective to accomplish.
            agents: List of available agent objects with role, goal, and tools.
            context: Optional additional context for planning decisions.

        Returns:
            A formatted planner prompt requesting an XML execution plan.

        Example:
            >>> template = PromptTemplate()
            >>> prompt = template.render_planner(
            ...     objective="Create a marketing campaign",
            ...     agents=[
            ...         {"role": "Writer", "goal": "Create content", "tools": []},
            ...         {"role": "Designer", "goal": "Create visuals", "tools": []}
            ...     ],
            ...     context="Target audience is tech professionals"
            ... )
        """
        return self.render(
            self.PLANNER_TEMPLATE,
            objective=objective,
            agents=agents,
            context=context,
        )

    def render_step_execution(
        self,
        action: str,
        description: str,
        arguments: dict | None = None,
        context: str = "",
    ) -> str:
        """Render a step execution prompt for planned workflow execution.

        Generates a prompt for executing a single step within a larger
        planned workflow, including action details, arguments, and context
        from previous steps.

        Args:
            action: The specific action to perform (e.g., "research", "write", "analyze").
            description: Detailed description of what this step should accomplish.
            arguments: Optional dictionary of arguments for the action.
            context: Optional context from previous steps in the workflow.

        Returns:
            A formatted step execution prompt.

        Example:
            >>> template = PromptTemplate()
            >>> prompt = template.render_step_execution(
            ...     action="analyze",
            ...     description="Analyze the competitor data",
            ...     arguments={"data_source": "market_report.csv"},
            ...     context="Previous step gathered competitor list"
            ... )
        """
        return self.render(
            self.STEP_EXECUTION_TEMPLATE,
            action=action,
            description=description,
            arguments=arguments or {},
            context=context,
        )

    def create_custom_template(self, template_string: str) -> Template | None:
        """Create a custom Jinja2 template from a template string.

        Creates a reusable Jinja2 Template object that can be rendered
        multiple times with different context variables.

        Args:
            template_string: A Jinja2 template string with variables and logic.

        Returns:
            A Jinja2 Template object if Jinja2 is enabled, None otherwise.

        Example:
            >>> template_engine = PromptTemplate()
            >>> custom = template_engine.create_custom_template(
            ...     "Report for {{ department }}:\\n{{ content }}"
            ... )
            >>> if custom:
            ...     result = custom.render(department="Sales", content="...")
        """
        if not self.use_jinja:
            return None
        return self.env.from_string(template_string)

    def get_template_variables(self, template_string: str) -> set:
        """Extract all variable names from a Jinja2 template string.

        Parses the template and identifies all undeclared variables that
        need to be provided when rendering the template.

        Args:
            template_string: A Jinja2 template string to analyze.

        Returns:
            A set of variable names found in the template. Returns an empty
            set if Jinja2 is disabled.

        Example:
            >>> template = PromptTemplate()
            >>> variables = template.get_template_variables(
            ...     "Hello {{ name }}, your order #{{ order_id }} is ready."
            ... )
            >>> print(variables)
            {'name', 'order_id'}
        """
        if not self.use_jinja:
            return set()

        ast = self.env.parse(template_string)
        return meta.find_undeclared_variables(ast)
