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
"""Agent definitions and built-in agents for the Xerxes framework.

This module exports built-in agent instances (e.g., code_agent, research_agent),
agent specification loaders, compaction utilities, and sub-agent management
primitives used throughout the system.

Main exports:
    - Built-in agents: code_agent, data_analyst_agent, planner_agent, research_agent
    - CompactionAgent and create_compaction_agent for context compaction
    - AgentDefinition and related loader functions
    - SubAgentManager and SubAgentTask for sub-agent orchestration
"""

from . import compaction_agent
from ._coder_agent import code_agent
from ._data_analyst_agent import data_analyst_agent
from ._planner_agent import planner_agent
from ._researcher_agent import research_agent
from .compaction_agent import CompactionAgent, create_compaction_agent
from .definitions import (
    BUILTIN_AGENTS,
    AgentDefinition,
    get_agent_definition,
    list_agent_definition_load_errors,
    list_agent_definitions,
    load_agent_definitions,
)
from .subagent_manager import SubAgentManager, SubAgentTask

__all__ = (
    "BUILTIN_AGENTS",
    "AgentDefinition",
    "CompactionAgent",
    "SubAgentManager",
    "SubAgentTask",
    "code_agent",
    "compaction_agent",
    "create_compaction_agent",
    "data_analyst_agent",
    "get_agent_definition",
    "list_agent_definition_load_errors",
    "list_agent_definitions",
    "load_agent_definitions",
    "planner_agent",
    "research_agent",
)
