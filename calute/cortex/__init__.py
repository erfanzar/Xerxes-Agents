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


"""
Cortex: A orchestration framework built on top of Calute.
Provides multi-agent collaboration with tasks, tools, and various execution strategies.
"""

from .agent import CortexAgent
from .cortex import Cortex, CortexOutput, MemoryConfig
from .dynamic import DynamicCortex, DynamicTaskBuilder, create_dynamic_cortex
from .enums import ChainType, ProcessType
from .memory_integration import CortexMemory
from .planner import CortexPlanner, ExecutionPlan, PlanStep
from .task import ChainLink, CortexTask, CortexTaskOutput
from .task_creator import TaskCreationPlan, TaskCreator, TaskDefinition
from .tool import CortexTool
from .universal_agent import UniversalAgent, UniversalTaskCreator

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
