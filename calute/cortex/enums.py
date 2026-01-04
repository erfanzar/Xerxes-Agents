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


"""Enumerations for Cortex framework.

This module provides enumeration types used throughout the Cortex
multi-agent orchestration framework. These enums define:
- Process execution types for agent orchestration strategies
- Chain types for task dependency structures

These enumerations enable type-safe configuration of agent workflows
and ensure consistent behavior across the Cortex framework.

Example:
    >>> from calute.cortex.enums import ProcessType, ChainType
    >>> process = ProcessType.SEQUENTIAL
    >>> chain = ChainType.LINEAR
"""

from enum import Enum


class ProcessType(Enum):
    """Enumeration of execution process types for Cortex orchestration.

    Defines the available orchestration strategies for multi-agent task
    execution within the Cortex framework. Each process type determines
    how agents coordinate and execute tasks.

    Attributes:
        SEQUENTIAL: Tasks execute one after another in order.
            Each task waits for the previous one to complete before starting.
        HIERARCHICAL: A manager agent delegates tasks to worker agents.
            The manager coordinates assignments and reviews outputs.
        PARALLEL: Multiple tasks execute simultaneously.
            Independent tasks can run concurrently for faster completion.
        CONSENSUS: Multiple agents work on the same task.
            Outputs are synthesized into a unified consensus response.
        PLANNED: Tasks follow an AI-generated execution plan.
            A planner agent creates a detailed step-by-step workflow.
    """

    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
    PARALLEL = "parallel"
    CONSENSUS = "consensus"
    PLANNED = "planned"


class ChainType(Enum):
    """Enumeration of chain types for task dependency structures.

    Defines how tasks are connected and depend on each other within
    a workflow. Chain types determine the flow and structure of
    task execution in complex multi-step processes.

    Attributes:
        LINEAR: Tasks form a straight sequence with single dependencies.
            Each task depends on exactly one predecessor.
        BRANCHING: Tasks can split into multiple parallel paths.
            One task can lead to multiple dependent tasks executing in parallel.
        LOOP: Tasks can form cycles for iterative processing.
            Allows repetition of task sequences until a condition is met.
    """

    LINEAR = "linear"
    BRANCHING = "branching"
    LOOP = "loop"
