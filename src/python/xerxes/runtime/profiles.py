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


"""Prompt profiles for controlling system prompt verbosity.

Profiles allow sub-agents and internal delegation to receive compressed
system prompts, reducing token usage while preserving safety-relevant
context (sandbox, guardrails).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PromptProfile(Enum):
    """System prompt verbosity profile.

    - FULL: all sections expanded (default, current behaviour).
    - COMPACT: compressed for sub-agents; trims workspace/bootstrap
      and caps skill instructions and tool lists.
    - MINIMAL: bare-minimum for internal delegation; only sandbox,
      guardrails, and a short tool list are included.
    - NONE: OpenClaw-style identity-only prompt with no runtime
      sections. Useful when the caller wants to supply all context.
    """

    FULL = "full"
    COMPACT = "compact"
    MINIMAL = "minimal"
    NONE = "none"


@dataclass
class PromptProfileConfig:
    """Fine-grained control over which prompt sections are emitted.

    Each flag controls whether the corresponding ``PromptContext``
    section is populated.  Length caps (``max_skill_instructions_length``,
    ``max_tools_listed``) truncate the content when set.
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
    """Return the canonical ``PromptProfileConfig`` for *profile*.

    The returned configs are:

    - **FULL** -- everything enabled, no caps.
    - **COMPACT** -- runtime info and safety sections kept; workspace
      and bootstrap dropped; skill instructions capped at 500 chars;
      tool list capped at 20 entries.
    - **MINIMAL** -- only sandbox, guardrails, and a 10-entry tool list.
    - **NONE** -- no runtime sections; the prompt builder returns only
      the base identity line.
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
