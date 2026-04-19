# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration and shared constants for Xerxes operator tooling.

Defines the three operator tool-name sets
(:data:`SAFE_OPERATOR_TOOLS`, :data:`HIGH_POWER_OPERATOR_TOOLS`, and
:data:`ALL_OPERATOR_TOOLS`) used by the policy engine to gate tool
access, as well as the :class:`OperatorRuntimeConfig` dataclass that
holds all runtime-tuneable knobs.
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
"""Operator tools that are always allowed, regardless of the
``power_tools_enabled`` flag.  These are low-risk, read-only or
informational tools."""

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
"""Operator tools that require explicit ``power_tools_enabled`` to be
active.  They can execute shell commands, modify files, spawn agents,
or interact with external services."""

ALL_OPERATOR_TOOLS: frozenset[str] = SAFE_OPERATOR_TOOLS | HIGH_POWER_OPERATOR_TOOLS
"""Union of safe and high-power operator tool names."""


@dataclass
class OperatorRuntimeConfig:
    """Opt-in runtime configuration for operator-style tools.

    Encapsulates every knob that controls how the operator subsystem
    behaves: whether it is enabled at all, which power level is active,
    browser and shell defaults, and sub-agent profile settings.

    Attributes:
        enabled: Master switch that activates operator tooling in the
            runtime.  When ``False``, none of the operator tools are
            registered.
        power_tools_enabled: When ``True``, high-power tools (shell,
            patch, browser navigation, sub-agents) are made available.
            Defaults to ``True`` so newly created agents and spawned
            sub-agents can use the full operator toolset unless a
            caller opts into a narrower policy.
        browser_headless: Whether the Playwright browser runs without a
            visible window.
        browser_screenshot_dir: Optional directory for browser
            screenshots.  When ``None``, temporary directories are used.
        shell_default_workdir: Default working directory for new PTY
            sessions.  ``None`` means the process working directory.
        shell_default_yield_ms: Default milliseconds to wait for
            initial PTY output before returning.
        shell_default_max_output_chars: Default maximum characters
            captured per PTY read operation.
        subagent_default_profile: Default prompt profile applied to
            newly spawned sub-agents.
        subagent_default_timeout_ms: Default timeout in milliseconds
            when waiting for sub-agent completion.
        allowed_tool_names: Set of tool names the policy engine should
            permit.  Defaults to :data:`ALL_OPERATOR_TOOLS`.
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
