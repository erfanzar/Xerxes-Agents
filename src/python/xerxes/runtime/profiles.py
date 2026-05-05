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
"""Profiles module for Xerxes.

Exports:
    - PromptProfile
    - PromptProfileConfig
    - get_profile_config"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PromptProfile(Enum):
    """Prompt profile.

    Inherits from: Enum
    """

    FULL = "full"
    COMPACT = "compact"
    MINIMAL = "minimal"
    NONE = "none"


@dataclass
class PromptProfileConfig:
    """Prompt profile config.

    Attributes:
        profile (PromptProfile): profile.
        include_runtime_info (bool): include runtime info.
        include_workspace_info (bool): include workspace info.
        include_sandbox_info (bool): include sandbox info.
        include_skills_index (bool): include skills index.
        include_enabled_skills (bool): include enabled skills.
        include_tools_list (bool): include tools list.
        include_guardrails (bool): include guardrails.
        include_bootstrap (bool): include bootstrap.
        include_relevant_memories (bool): include relevant memories.
        include_user_profile (bool): include user profile.
        max_skill_instructions_length (int | None): max skill instructions length.
        max_tools_listed (int | None): max tools listed.
        max_memories_injected (int): max memories injected."""

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
    """Retrieve the profile config.

    Args:
        profile (PromptProfile): IN: profile. OUT: Consumed during execution.
    Returns:
        PromptProfileConfig: OUT: Result of the operation."""

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
