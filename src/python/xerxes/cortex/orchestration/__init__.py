# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.

"""Orchestration sub-package for the Cortex multi-agent framework.

This sub-package provides the high-level orchestration components for
coordinating multiple AI agents to accomplish complex tasks. It contains
the main orchestrator, task definitions, planning engine, task creator,
and dynamic execution capabilities.

Modules:
    cortex: The main Cortex orchestrator class coordinating agents and
        tasks through sequential, parallel, hierarchical, consensus, and
        planned execution strategies. Also provides CortexOutput for
        structured results and MemoryConfig for memory configuration.
    task: CortexTask for defining executable work units with output
        validation, retry logic, dependency management, and chaining.
        Includes CortexTaskOutput for rich result metadata and ChainLink
        for conditional task routing.
    planner: CortexPlanner for AI-powered XML-based execution plan
        generation and step-by-step execution with dependency resolution.
        Provides PlanStep and ExecutionPlan data structures.
    task_creator: TaskCreator for automatically generating structured
        task breakdowns from natural language objectives. Provides
        TaskDefinition and TaskCreationPlan for task specifications.
    dynamic: DynamicCortex extending Cortex with runtime task creation
        from natural language prompts, and DynamicTaskBuilder for
        on-the-fly task generation and chaining.

Example:
    >>> from xerxes.cortex.orchestration import Cortex, CortexTask, CortexOutput
    >>> cortex = Cortex(agents=[agent], tasks=[task], llm=llm)
    >>> result = cortex.kickoff()
    >>> print(result.raw_output)
"""

from .cortex import Cortex, CortexOutput, MemoryConfig
from .dynamic import DynamicCortex, DynamicTaskBuilder, create_dynamic_cortex
from .planner import CortexPlanner, ExecutionPlan, PlanStep
from .task import ChainLink, CortexTask, CortexTaskOutput
from .task_creator import TaskCreationPlan, TaskCreator, TaskDefinition

__all__ = [
    "ChainLink",
    "Cortex",
    "CortexOutput",
    "CortexPlanner",
    "CortexTask",
    "CortexTaskOutput",
    "DynamicCortex",
    "DynamicTaskBuilder",
    "ExecutionPlan",
    "MemoryConfig",
    "PlanStep",
    "TaskCreationPlan",
    "TaskCreator",
    "TaskDefinition",
    "create_dynamic_cortex",
]
