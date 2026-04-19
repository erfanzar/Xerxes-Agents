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


"""Cortex: A multi-agent orchestration framework built on top of Xerxes.

This module provides a comprehensive orchestration framework for building and managing
complex multi-agent systems with sophisticated collaboration patterns, task management,
and execution strategies. Cortex enables agents to work together on complex workflows
with support for sequential, parallel, hierarchical, consensus-based, and planned
execution modes.

Key Features:
    - Multi-agent orchestration with dynamic task assignment
    - Multiple execution strategies (sequential, parallel, hierarchical, consensus, planned)
    - Task chaining and conditional workflows
    - Integrated memory system for context preservation
    - Tool integration for agent capabilities
    - Streaming support for real-time output
    - Dynamic task and agent creation
    - Universal agents for flexible task handling

Components:
    - Cortex: Main orchestrator for coordinating agents and tasks
    - CortexAgent: Intelligent agent with specific role, goal, and capabilities
    - CortexTask: Task definition with execution context and dependencies
    - CortexTool: Tool wrapper for agent function integration
    - CortexMemory: Memory management for context and knowledge retention
    - CortexPlanner: AI-powered planning for complex task sequences
    - DynamicCortex: Runtime configuration for dynamic workflows

Example:
    >>> from xerxes.cortex import Cortex, CortexAgent, CortexTask, ProcessType
    >>> from xerxes.llms import OpenAILLM
    >>>
    >>> llm = OpenAILLM(api_key="your-api-key")
    >>>
    >>> researcher = CortexAgent(
    ...     role="Research Analyst",
    ...     goal="Gather and analyze information",
    ...     backstory="Expert in data research and analysis"
    ... )
    >>>
    >>> task = CortexTask(
    ...     description="Research market trends for AI",
    ...     expected_output="Comprehensive market analysis report",
    ...     agent=researcher
    ... )
    >>>
    >>> cortex = Cortex(
    ...     agents=[researcher],
    ...     tasks=[task],
    ...     llm=llm,
    ...     process=ProcessType.SEQUENTIAL
    ... )
    >>>
    >>> result = cortex.kickoff()
    >>> print(result.raw_output)
"""

from .agents.agent import CortexAgent
from .agents.memory_integration import CortexMemory
from .agents.universal_agent import UniversalAgent, UniversalTaskCreator
from .core.enums import ChainType, ProcessType
from .core.tool import CortexTool
from .orchestration.cortex import Cortex, CortexOutput, MemoryConfig
from .orchestration.dynamic import DynamicCortex, DynamicTaskBuilder, create_dynamic_cortex
from .orchestration.planner import CortexPlanner, ExecutionPlan, PlanStep
from .orchestration.task import ChainLink, CortexTask, CortexTaskOutput
from .orchestration.task_creator import TaskCreationPlan, TaskCreator, TaskDefinition

__all__ = [
    "ChainLink",
    "ChainType",
    "Cortex",
    "CortexAgent",
    "CortexMemory",
    "CortexOutput",
    "CortexPlanner",
    "CortexTask",
    "CortexTaskOutput",
    "CortexTool",
    "DynamicCortex",
    "DynamicTaskBuilder",
    "ExecutionPlan",
    "MemoryConfig",
    "PlanStep",
    "ProcessType",
    "TaskCreationPlan",
    "TaskCreator",
    "TaskDefinition",
    "UniversalAgent",
    "UniversalTaskCreator",
    "create_dynamic_cortex",
]
