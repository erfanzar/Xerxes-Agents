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
"""Prompt template structure and section enumeration.

Defines ``PromptSection`` — an enum of standard prompt parts — and
``PromptTemplate`` — a dataclass that orders and stores those sections.
"""

from dataclasses import dataclass
from enum import Enum

SEP = "  "


class PromptSection(Enum):
    """Standard sections of a Xerxes prompt template."""

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
    """Ordered collection of prompt sections with default labels."""

    sections: dict[PromptSection, str] | None = None
    section_order: list[PromptSection] | None = None

    def __post_init__(self):
        """Initialize defaults for ``sections`` and ``section_order``."""
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
