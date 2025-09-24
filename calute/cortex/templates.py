
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

"""Jinja2 template support for Cortex prompts"""

from jinja2 import Environment, Template, meta


class PromptTemplate:
    """Prompt template with Jinja2 support"""

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
        self.use_jinja = True
        self.env = Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False)

    def render(self, template: str, **kwargs) -> str:
        """Render a template with the given context"""
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
        """Render agent prompt"""
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
        """Render task prompt"""
        return self.render(
            self.TASK_TEMPLATE,
            description=description,
            expected_output=expected_output,
            context=context,
            constraints=constraints,
        )

    def render_manager_delegation(self, agents: list, tasks: list) -> str:
        """Render manager delegation prompt"""
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
        """Render manager review prompt"""
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
        """Render consensus prompt"""
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
        """Render planner prompt for XML plan generation"""
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
        """Render step execution prompt"""
        return self.render(
            self.STEP_EXECUTION_TEMPLATE,
            action=action,
            description=description,
            arguments=arguments or {},
            context=context,
        )

    def create_custom_template(self, template_string: str) -> Template | None:
        """Create a custom Jinja2 template"""
        if not self.use_jinja:
            return None
        return self.env.from_string(template_string)

    def get_template_variables(self, template_string: str) -> set:
        """Extract variables from a template string"""
        if not self.use_jinja:
            return set()

        ast = self.env.parse(template_string)
        return meta.find_undeclared_variables(ast)
