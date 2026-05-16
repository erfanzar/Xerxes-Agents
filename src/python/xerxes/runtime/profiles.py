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
"""Prompt-profile presets that gate :class:`PromptContextBuilder` sections.

Four profiles ship out of the box: ``FULL`` (everything), ``COMPACT``
(workspace/bootstrap dropped, skill/tools capped), ``MINIMAL`` (only
sandbox + tools + guardrails), and ``NONE`` (identity line only).
:func:`get_profile_config` resolves a :class:`PromptProfile` to a fully
populated :class:`PromptProfileConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PromptProfile(Enum):
    """Named prompt-prefix verbosity level.

    Attributes:
        FULL: Every section enabled (default).
        COMPACT: Drop workspace/bootstrap; cap skills and tool list size.
        MINIMAL: Only sandbox, tool list (≤10), and guardrails.
        NONE: Bare identity line; for sub-agents that need no context.
    """

    FULL = "full"
    COMPACT = "compact"
    MINIMAL = "minimal"
    NONE = "none"


@dataclass
class PromptProfileConfig:
    """Section gating and size caps for one :class:`PromptProfile`.

    Attributes:
        profile: Originating profile enum value.
        include_runtime_info: Whether to render the runtime/datetime sections.
        include_workspace_info: Whether to render the workspace section.
        include_sandbox_info: Whether to render the sandbox section.
        include_skills_index: Whether to render the skills index.
        include_enabled_skills: Whether to render enabled skill instructions.
        include_tools_list: Whether to render the tools bullet list.
        include_guardrails: Whether to render the guardrails section.
        include_bootstrap: Whether to invoke the ``bootstrap_files`` hook.
        include_relevant_memories: Whether to inject memory snippets.
        include_user_profile: Whether to render the user-profile blurb.
        max_skill_instructions_length: Hard cap on per-skill instruction text.
        max_tools_listed: Hard cap on the tool list size.
        max_memories_injected: Maximum memory snippets requested per turn.
    """

    profile: PromptProfile = PromptProfile.FULL
    include_runtime_info: bool = True
    include_workspace_info: bool = True
    include_sandbox_info: bool = True
    include_skills_index: bool = True
    include_enabled_skills: bool = True
    include_tools_list: bool = True
    include_guardrails: bool = True
    include_bootstrap: bool = True
    include_relevant_memories: bool = True
    include_user_profile: bool = True
    max_skill_instructions_length: int | None = None
    max_tools_listed: int | None = None
    max_memories_injected: int = 5


def get_profile_config(profile: PromptProfile) -> PromptProfileConfig:
    """Return the canonical :class:`PromptProfileConfig` preset for ``profile``.

    Raises:
        ValueError: ``profile`` is not one of the four supported enum values.
    """

    if profile == PromptProfile.FULL:
        return PromptProfileConfig(profile=PromptProfile.FULL)

    if profile == PromptProfile.COMPACT:
        return PromptProfileConfig(
            profile=PromptProfile.COMPACT,
            include_runtime_info=True,
            include_workspace_info=False,
            include_sandbox_info=True,
            include_skills_index=True,
            include_enabled_skills=True,
            include_tools_list=True,
            include_guardrails=True,
            include_bootstrap=False,
            max_skill_instructions_length=500,
            max_tools_listed=20,
        )

    if profile == PromptProfile.MINIMAL:
        return PromptProfileConfig(
            profile=PromptProfile.MINIMAL,
            include_runtime_info=False,
            include_workspace_info=False,
            include_sandbox_info=True,
            include_skills_index=False,
            include_enabled_skills=False,
            include_tools_list=True,
            include_guardrails=True,
            include_bootstrap=False,
            include_relevant_memories=False,
            include_user_profile=False,
            max_tools_listed=10,
        )

    if profile == PromptProfile.NONE:
        return PromptProfileConfig(
            profile=PromptProfile.NONE,
            include_runtime_info=False,
            include_workspace_info=False,
            include_sandbox_info=False,
            include_skills_index=False,
            include_enabled_skills=False,
            include_tools_list=False,
            include_guardrails=False,
            include_bootstrap=False,
            include_relevant_memories=False,
            include_user_profile=False,
        )

    raise ValueError(f"Unknown profile: {profile!r}")
