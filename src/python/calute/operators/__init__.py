# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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

"""Operator tooling for Calute runtime.

This package provides the operator subsystem that powers interactive,
long-running tool execution within Calute.  It includes:

- **Browser automation** via a Playwright-backed manager.
- **Persistent PTY sessions** for interactive shell commands.
- **Sub-agent spawning** and lifecycle management.
- **Plan state tracking** for structured multi-step execution.
- **User prompt management** for runtime clarification questions.
- **Configuration** constants that classify tools into safe and
  high-power categories.

Typical usage is through :class:`OperatorState`, which composes all
sub-managers and exposes callable operator tools to the Calute runtime.
"""

from .browser import BrowserManager, BrowserPageState
from .config import (
    ALL_OPERATOR_TOOLS,
    HIGH_POWER_OPERATOR_TOOLS,
    SAFE_OPERATOR_TOOLS,
    OperatorRuntimeConfig,
)
from .pty import PTYSessionManager
from .state import OperatorState
from .subagents import SpawnedAgentManager
from .types import ImageInspectionResult, OperatorPlanState, OperatorPlanStep, PendingUserPrompt, UserPromptOption
from .user_prompt import UserPromptManager

__all__ = (
    "ALL_OPERATOR_TOOLS",
    "HIGH_POWER_OPERATOR_TOOLS",
    "SAFE_OPERATOR_TOOLS",
    "BrowserManager",
    "BrowserPageState",
    "ImageInspectionResult",
    "OperatorPlanState",
    "OperatorPlanStep",
    "OperatorRuntimeConfig",
    "OperatorState",
    "PTYSessionManager",
    "PendingUserPrompt",
    "SpawnedAgentManager",
    "UserPromptManager",
    "UserPromptOption",
)
