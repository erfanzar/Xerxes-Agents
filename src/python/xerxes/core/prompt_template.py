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
"""Ordered prompt sections used by the legacy template builder.

Defines :class:`PromptSection` (an enum of the canonical pieces a system
prompt can have) and :class:`PromptTemplate` (a dataclass holding the
per-section header labels and the rendering order). The newer streaming
loop builds prompts directly, but legacy modules still emit them through
this template for parity.
"""

from dataclasses import dataclass
from enum import Enum

SEP = "  "


class PromptSection(Enum):
    """Canonical section names used to slot content into a prompt template."""

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
    """Per-section header labels plus the order they should be rendered.

    Attributes:
        sections: Map from :class:`PromptSection` to its header text.
        section_order: Order to render sections in the final prompt.
    """

    sections: dict[PromptSection, str] | None = None
    section_order: list[PromptSection] | None = None

    def __post_init__(self):
        """Populate ``sections`` and ``section_order`` with sensible defaults if missing."""
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
