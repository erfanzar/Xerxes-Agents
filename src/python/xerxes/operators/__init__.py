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
"""Operator subsystem — session-attached managers and high-power tools.

Re-exports the public surface of the operator package:

* :class:`OperatorState` and :class:`OperatorRuntimeConfig` — the session
  integration point.
* Manager classes (:class:`PTYSessionManager`, :class:`BrowserManager`,
  :class:`SpawnedAgentManager`, :class:`UserPromptManager`) — direct access
  for callers that want to bypass the tool layer.
* Value objects exchanged on the wire (:class:`ImageInspectionResult`,
  :class:`PendingUserPrompt`, :class:`OperatorPlanState`, ...).
* Tool name registries (:data:`SAFE_OPERATOR_TOOLS`,
  :data:`HIGH_POWER_OPERATOR_TOOLS`, :data:`ALL_OPERATOR_TOOLS`).
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
