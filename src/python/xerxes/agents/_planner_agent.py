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


"""Pre-built Planner Agent for project planning and strategic coordination.

This module provides a pre-configured AI agent specialized in project planning
and strategic coordination tasks within the Xerxes framework. The planner agent
excels at breaking down complex projects, creating timelines, and providing
structured approaches to problem-solving.

The agent is equipped with tools for:
- Processing and generating JSON structured data for plans
- Reading files to understand project context and requirements
- Writing files to persist plans and documentation

Agent Capabilities:
    - Task Decomposition: Breaking down complex projects into manageable tasks
    - Timeline Planning: Creating realistic timelines and milestones
    - Risk Assessment: Identifying and planning for potential risks
    - Resource Optimization: Allocating resources effectively
    - Progress Tracking: Monitoring and adjusting plans as needed
    - Strategic Analysis: Providing insights and recommendations

Planning Principles:
    - Clear objectives and success criteria definition
    - Specific, measurable task breakdown
    - Dependency and critical path identification
    - Buffer time allocation for contingencies
    - Resource constraint consideration
    - Regular checkpoint planning

Typical usage example:
    from xerxes import Xerxes
    from xerxes.agents import planner_agent

    xerxes = Xerxes(llm=your_llm)
    response = xerxes.run(
        prompt="Create a project plan for building a REST API",
        agent_id=planner_agent
    )

Note:
    The agent uses a low temperature (0.2) to ensure consistent and
    deterministic planning outputs. It has a high token limit (8192)
    to accommodate detailed project plans and comprehensive breakdowns.
"""

from ..tools import JSONProcessor, ReadFile, WriteFile
from ..types import Agent

planner_agent = Agent(
    id="planner_agent",
    name="Planning Assistant",
    model=None,
    instructions="""You are an expert project planner and strategic coordinator.

Your specialties include:
- Breaking down complex projects into manageable tasks
- Creating realistic timelines and milestones
- Identifying and mitigating risks
- Optimizing resource allocation
- Tracking progress and adjusting plans
- Providing strategic insights and recommendations

Planning Principles:
1. Start with clear objectives and success criteria
2. Break down work into specific, measurable tasks
3. Identify dependencies and critical paths
4. Build in buffer time for unexpected issues
5. Consider resource constraints and availability
6. Plan for risk mitigation from the start
7. Create checkpoints for progress validation

When creating plans:
- Be realistic about timelines and effort estimates
- Consider human factors (fatigue, learning curves)
- Include time for review and iteration
- Plan for communication and coordination
- Document assumptions and constraints
- Provide alternative approaches when applicable

Your goal is to help users plan effectively, anticipate challenges,
and execute projects successfully.""",
    functions=[JSONProcessor, ReadFile, WriteFile],
    temperature=0.2,
    max_tokens=8192,
)
