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
"""Config module for Xerxes.

Exports:
    - OperatorRuntimeConfig"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..runtime.profiles import PromptProfile

SAFE_OPERATOR_TOOLS: frozenset[str] = frozenset(
    {
        "ask_user",
        "web.time",
        "update_plan",
    }
)

HIGH_POWER_OPERATOR_TOOLS: frozenset[str] = frozenset(
    {
        "exec_command",
        "write_stdin",
        "apply_patch",
        "spawn_agent",
        "resume_agent",
        "send_input",
        "wait_agent",
        "close_agent",
        "view_image",
        "web.search_query",
        "web.image_query",
        "web.open",
        "web.click",
        "web.find",
        "web.screenshot",
        "web.weather",
        "web.finance",
        "web.sports",
    }
)

ALL_OPERATOR_TOOLS: frozenset[str] = SAFE_OPERATOR_TOOLS | HIGH_POWER_OPERATOR_TOOLS


@dataclass
class OperatorRuntimeConfig:
    """Operator runtime config.

    Attributes:
        enabled (bool): enabled.
        power_tools_enabled (bool): power tools enabled.
        browser_headless (bool): browser headless.
        browser_screenshot_dir (str | None): browser screenshot dir.
        shell_default_workdir (str | None): shell default workdir.
        shell_default_yield_ms (int): shell default yield ms.
        shell_default_max_output_chars (int): shell default max output chars.
        subagent_default_profile (PromptProfile | str): subagent default profile.
        subagent_default_timeout_ms (int): subagent default timeout ms.
        allowed_tool_names (set[str]): allowed tool names."""

    enabled: bool = False
    power_tools_enabled: bool = True
    browser_headless: bool = True
    browser_screenshot_dir: str | None = None
    shell_default_workdir: str | None = None
    shell_default_yield_ms: int = 1000
    shell_default_max_output_chars: int = 4000
    subagent_default_profile: PromptProfile | str = PromptProfile.MINIMAL
    subagent_default_timeout_ms: int = 30000
    allowed_tool_names: set[str] = field(default_factory=lambda: set(ALL_OPERATOR_TOOLS))
