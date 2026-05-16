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
"""Runtime configuration and tool registries for the operator subsystem.

Provides the canonical tool-name sets that distinguish *safe* helpers from
*high-power* operator capabilities (shell exec, patch application, subagent
spawning, browser control) plus :class:`OperatorRuntimeConfig`, the
session-scoped dataclass that controls which operator features are wired
into the streaming runtime.
"""

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
    """Session-scoped knobs that govern operator behaviour.

    A single instance is owned by the daemon-side session and forwarded to
    every operator manager during initialisation. Values flow into PTY
    bookkeeping, browser provider construction, and subagent defaults.

    Attributes:
        enabled: Master switch — when ``False`` the operator subsystem stays
            dormant and only the bare conversational tools are registered.
        power_tools_enabled: Permit destructive / privileged tools such as
            ``exec_command``, ``apply_patch`` and ``spawn_agent``.
        browser_headless: Run the browser provider without a visible window.
        browser_screenshot_dir: Directory for browser screenshots; ``None``
            falls back to the provider default.
        shell_default_workdir: Default working directory injected into new
            PTY sessions when the caller omits one.
        shell_default_yield_ms: Default soft deadline (ms) used by
            ``exec_command`` before yielding partial output.
        shell_default_max_output_chars: Default cap on captured shell output
            per yield to avoid flooding the wire.
        subagent_default_profile: Prompt profile applied to spawned
            subagents that don't request a specific profile.
        subagent_default_timeout_ms: Default timeout (ms) before a spawned
            subagent is considered unresponsive.
        allowed_tool_names: Effective allow-list of operator tools. The
            streaming runtime intersects this with the requested tool set.
    """

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
