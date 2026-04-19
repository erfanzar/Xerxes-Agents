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


"""Security subsystem for Xerxes.

Provides tool policy enforcement, sandbox execution configuration,
and sandbox backend implementations.
"""

from .policy import PolicyAction, PolicyEngine, ToolPolicy, ToolPolicyViolation
from .sandbox import (
    ExecutionContext,
    ExecutionDecision,
    SandboxBackend,
    SandboxBackendConfig,
    SandboxConfig,
    SandboxExecutionUnavailableError,
    SandboxMode,
    SandboxRouter,
)

__all__ = [
    "ExecutionContext",
    "ExecutionDecision",
    "PolicyAction",
    "PolicyEngine",
    "SandboxBackend",
    "SandboxBackendConfig",
    "SandboxConfig",
    "SandboxExecutionUnavailableError",
    "SandboxMode",
    "SandboxRouter",
    "ToolPolicy",
    "ToolPolicyViolation",
]
