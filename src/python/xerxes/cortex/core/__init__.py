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


"""Core sub-package for the Cortex multi-agent orchestration framework.

This sub-package provides foundational building blocks used throughout the
Cortex framework, including enumeration types, tool wrappers, string
utilities, and prompt template engines.

Modules:
    enums: ProcessType and ChainType enumerations defining execution
        strategies (sequential, parallel, hierarchical, consensus, planned)
        and task dependency structures (linear, branching, loop).
    string_utils: Template variable interpolation, extraction, and
        validation utilities using simple ``{variable}`` placeholder syntax.
    templates: Jinja2-based PromptTemplate engine with pre-defined templates
        for agent prompts, task prompts, manager delegation, consensus
        synthesis, strategic planning, and step execution.
    tool: CortexTool dataclass for wrapping Python callables as agent tools
        with automatic OpenAI-compatible JSON schema generation.

Example:
    >>> from xerxes.cortex.core import ProcessType, CortexTool, PromptTemplate
    >>> tool = CortexTool.from_function(my_function, name="search")
    >>> template = PromptTemplate()
    >>> prompt = template.render_agent_prompt(
    ...     role="Analyst", goal="Analyze data", backstory="Expert"
    ... )
"""

from .enums import ChainType, ProcessType
from .string_utils import extract_template_variables, interpolate_inputs, validate_inputs_for_template
from .templates import PromptTemplate
from .tool import CortexTool

__all__ = [
    "ChainType",
    "CortexTool",
    "ProcessType",
    "PromptTemplate",
    "extract_template_variables",
    "interpolate_inputs",
    "validate_inputs_for_template",
]
