# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.


"""Prompt template structures for Xerxes agents.

Provides :class:`PromptSection` (an enum of standard prompt sections) and
:class:`PromptTemplate` (a configurable dataclass for structuring agent
prompts with ordered, labelled sections).
"""

from dataclasses import dataclass
from enum import Enum

SEP = "  "


class PromptSection(Enum):
    """Enumeration of different sections in a structured prompt.

    This enum defines the standard sections that can be included in a
    structured prompt template, allowing for consistent prompt organization
    across different agents and use cases.

    Attributes:
        SYSTEM: System-level instructions and configuration.
        PERSONA: Agent personality and role definition.
        RULES: Behavioral rules and constraints for the agent.
        FUNCTIONS: Available function/tool definitions.
        TOOLS: Tool usage instructions and format specifications.
        EXAMPLES: Example interactions for few-shot learning.
        CONTEXT: Contextual information and variables.
        HISTORY: Conversation history from previous turns.
        PROMPT: The actual user prompt/query.

    Example:
        >>> template = PromptTemplate(
        ...     sections={PromptSection.SYSTEM: "INSTRUCTIONS:"},
        ...     section_order=[PromptSection.SYSTEM, PromptSection.PROMPT]
        ... )
    """

    SYSTEM = "system"
    PERSONA = "persona"
    RULES = "rules"
    FUNCTIONS = "functions"
    TOOLS = "tools"
    EXAMPLES = "examples"
    CONTEXT = "context"
    HISTORY = "history"
    PROMPT = "prompt"


@dataclass
class PromptTemplate:
    """Configurable template for structuring agent prompts.

    This class provides a flexible way to structure prompts with different
    sections that can be customized or reordered based on requirements.

    Attributes:
        sections: Dictionary mapping PromptSection enums to their header strings.
        section_order: List defining the order in which sections appear in the prompt.

    Example:
        >>> template = PromptTemplate(
        ...     sections={PromptSection.SYSTEM: "INSTRUCTIONS:"},
        ...     section_order=[PromptSection.SYSTEM, PromptSection.PROMPT]
        ... )
    """

    sections: dict[PromptSection, str] | None = None
    section_order: list[PromptSection] | None = None

    def __post_init__(self):
        """Initialize default sections and ordering if not provided.

        Sets up standard prompt sections with appropriate headers and
        establishes a default ordering that works well for most use cases.
        """
        self.sections = self.sections or {
            PromptSection.SYSTEM: "SYSTEM:",
            PromptSection.RULES: "RULES:",
            PromptSection.FUNCTIONS: "FUNCTIONS:",
            PromptSection.TOOLS: f"TOOLS:\n{SEP}When using tools, follow this format:",
            PromptSection.EXAMPLES: f"EXAMPLES:\n{SEP}",
            PromptSection.CONTEXT: "CONTEXT:\n",
            PromptSection.HISTORY: f"HISTORY:\n{SEP}Conversation so far:\n",
            PromptSection.PROMPT: "PROMPT:\n",
        }

        self.section_order = self.section_order or [
            PromptSection.SYSTEM,
            PromptSection.RULES,
            PromptSection.FUNCTIONS,
            PromptSection.TOOLS,
            PromptSection.EXAMPLES,
            PromptSection.CONTEXT,
            PromptSection.HISTORY,
            PromptSection.PROMPT,
        ]
