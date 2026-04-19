# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.

"""Agents sub-package for the Cortex multi-agent orchestration framework.

This sub-package provides the core agent abstractions used within the Cortex
framework for building intelligent, autonomous AI agents. It contains:

Modules:
    agent: The CortexAgent class for defining agents with roles, goals,
        capabilities, tool integration, delegation, and memory support.
    memory_integration: The CortexMemory class providing a unified memory
        system integrating short-term, long-term, entity, user, and
        contextual memory types for comprehensive context management.
    universal_agent: The UniversalAgent class extending CortexAgent with
        a pre-configured comprehensive tool set (web search, file ops,
        git, code analysis, Python execution), and UniversalTaskCreator
        for automatic task creation and agent assignment.

Example:
    >>> from xerxes.cortex.agents import CortexAgent, CortexMemory, UniversalAgent
    >>> agent = CortexAgent(
    ...     role="Data Analyst",
    ...     goal="Analyze data and provide insights",
    ...     backstory="Expert in statistical analysis"
    ... )
    >>> memory = CortexMemory(enable_short_term=True, enable_long_term=True)
    >>> universal = UniversalAgent(llm=my_llm, verbose=True)
"""

from .agent import CortexAgent
from .memory_integration import CortexMemory
from .universal_agent import UniversalAgent, UniversalTaskCreator

__all__ = [
    "CortexAgent",
    "CortexMemory",
    "UniversalAgent",
    "UniversalTaskCreator",
]
