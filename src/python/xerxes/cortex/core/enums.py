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
"""Enumeration types for Cortex process and chain configurations."""

from enum import Enum


class ProcessType(Enum):
    """Defines the execution strategy for a Cortex workflow.

    Attributes:
        SEQUENTIAL (str): Execute tasks one after another in order.
        HIERARCHICAL (str): Use a manager agent to delegate and review tasks.
        PARALLEL (str): Execute independent tasks concurrently.
        CONSENSUS (str): Aggregate outputs from multiple agents for each task.
        PLANNED (str): Generate and follow a structured execution plan.
    """

    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
    PARALLEL = "parallel"
    CONSENSUS = "consensus"
    PLANNED = "planned"


class ChainType(Enum):
    """Defines the linking behavior between tasks in a chain.

    Attributes:
        LINEAR (str): Proceed to the next task unconditionally.
        BRANCHING (str): Choose the next task based on a condition.
        LOOP (str): Repeat tasks based on a loop condition.
    """

    LINEAR = "linear"
    BRANCHING = "branching"
    LOOP = "loop"
