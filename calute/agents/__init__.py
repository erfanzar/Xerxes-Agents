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


"""Pre-built agents for specialized tasks within the Calute framework.

This module provides a collection of ready-to-use agent configurations optimized
for specific domains and use cases. These pre-built agents can be used directly
or serve as templates for creating custom specialized agents.

Available Agents:
    - code_agent: Specialized for code generation, review, and debugging tasks
    - data_analyst_agent: Optimized for data analysis and visualization tasks
    - planner_agent: Designed for task planning and project management
    - research_agent: Configured for research and information gathering
    - CompactionAgent: Agent for memory compaction and summarization

Example:
    >>> from calute.agents import code_agent, research_agent
    >>> from calute import OpenAILLM
    >>>
    >>> llm = OpenAILLM(api_key="your-api-key")
    >>> coder = code_agent(llm=llm)
    >>> response = coder.query("Write a Python function to sort a list")
"""

from . import compaction_agent
from ._coder_agent import code_agent
from ._data_analyst_agent import data_analyst_agent
from ._planner_agent import planner_agent
from ._researcher_agent import research_agent
from .compaction_agent import CompactionAgent, create_compaction_agent

__all__ = (
    "CompactionAgent",
    "code_agent",
    "compaction_agent",
    "create_compaction_agent",
    "data_analyst_agent",
    "planner_agent",
    "research_agent",
)
