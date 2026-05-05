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
"""Cortex package for multi-agent orchestration and task execution.

This module exports the main components of the Cortex system, including agents,
memory integration, orchestration engines, task management, and core utilities.
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
