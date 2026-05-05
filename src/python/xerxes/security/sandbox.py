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
"""Sandbox module for Xerxes.

Exports:
    - logger
    - SandboxMode
    - SandboxBackendConfig
    - SandboxConfig
    - ExecutionContext
    - ExecutionDecision
    - SandboxExecutionUnavailableError
    - SandboxBackend
    - SandboxRouter"""

from __future__ import annotations

import logging
import typing as tp
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SandboxMode(Enum):
    """Sandbox mode.

    Inherits from: Enum
    """

    OFF = "off"
    WARN = "warn"
    STRICT = "strict"


@dataclass
class SandboxBackendConfig:
    """Sandbox backend config.

    Attributes:
        image (str): image.
        mount_paths (dict[str, str]): mount paths.
        mount_readonly (bool): mount readonly.
        env_vars (dict[str, str]): env vars.
        extra_args (dict[str, tp.Any]): extra args."""

    image: str = "python:3.12-slim"
    mount_paths: dict[str, str] = field(default_factory=dict)
    mount_readonly: bool = True
    env_vars: dict[str, str] = field(default_factory=dict)
    extra_args: dict[str, tp.Any] = field(default_factory=dict)


@dataclass
class SandboxConfig:
    """Sandbox config.

    Attributes:
        mode (SandboxMode): mode.
        sandboxed_tools (set[str]): sandboxed tools.
        elevated_tools (set[str]): elevated tools.
        sandbox_timeout (float): sandbox timeout.
        sandbox_memory_limit_mb (int): sandbox memory limit mb.
        sandbox_network_access (bool): sandbox network access.
        working_directory (str | None): working directory.
        backend_type (str | None): backend type.
        backend_config (SandboxBackendConfig): backend config."""

    mode: SandboxMode = SandboxMode.OFF
    sandboxed_tools: set[str] = field(default_factory=set)
    elevated_tools: set[str] = field(default_factory=set)
    sandbox_timeout: float = 30.0
    sandbox_memory_limit_mb: int = 512
    sandbox_network_access: bool = False
    working_directory: str | None = None
    backend_type: str | None = None
    backend_config: SandboxBackendConfig = field(default_factory=SandboxBackendConfig)


class ExecutionContext(Enum):
    """Execution context.

    Inherits from: Enum
    """

    HOST = "host"
    SANDBOX = "sandbox"


@dataclass
class ExecutionDecision:
    """Execution decision.

    Attributes:
        context (ExecutionContext): context.
        tool_name (str): tool name.
        reason (str): reason."""

    context: ExecutionContext
    tool_name: str
    reason: str


class SandboxExecutionUnavailableError(RuntimeError):
    """Sandbox execution unavailable error.

    Inherits from: RuntimeError
    """

    def __init__(self, tool_name: str) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            tool_name (str): IN: tool name. OUT: Consumed during execution."""

        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' requires sandbox execution, but no sandbox backend is configured")


class SandboxBackend(tp.Protocol):
    """Sandbox backend.

    Inherits from: tp.Protocol
    """

    def execute(self, tool_name: str, func: tp.Callable, arguments: dict) -> tp.Any:
        """Execute.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            tool_name (str): IN: tool name. OUT: Consumed during execution.
            func (tp.Callable): IN: func. OUT: Consumed during execution.
            arguments (dict): IN: arguments. OUT: Consumed during execution.
        Returns:
            tp.Any: OUT: Result of the operation."""
        ...

    def is_available(self) -> bool:
        """Check whether available.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            bool: OUT: Result of the operation."""
        ...

    def get_capabilities(self) -> dict[str, tp.Any]:
        """Retrieve the capabilities.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""
        ...


class SandboxRouter:
    """Sandbox router."""

    def __init__(self, config: SandboxConfig | None = None, backend: SandboxBackend | None = None) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            config (SandboxConfig | None, optional): IN: config. Defaults to None. OUT: Consumed during execution.
            backend (SandboxBackend | None, optional): IN: backend. Defaults to None. OUT: Consumed during execution."""

        self.config = config or SandboxConfig()
        self.backend = backend

    def decide(self, tool_name: str) -> ExecutionDecision:
        """Decide.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            tool_name (str): IN: tool name. OUT: Consumed during execution.
        Returns:
            ExecutionDecision: OUT: Result of the operation."""

        if tool_name in self.config.elevated_tools:
            return ExecutionDecision(
                context=ExecutionContext.HOST,
                tool_name=tool_name,
                reason="Tool is marked as elevated",
            )

        if self.config.mode == SandboxMode.OFF:
            return ExecutionDecision(
                context=ExecutionContext.HOST,
                tool_name=tool_name,
                reason="Sandbox mode is off",
            )

        if tool_name in self.config.sandboxed_tools:
            if self.config.mode == SandboxMode.WARN:
                logger.warning("Tool '%s' would run in sandbox (mode=warn, running on host)", tool_name)
                return ExecutionDecision(
                    context=ExecutionContext.HOST,
                    tool_name=tool_name,
                    reason="Warn mode advisory: tool would run in sandbox, executing on host",
                )
            elif self.config.mode == SandboxMode.STRICT:
                return ExecutionDecision(
                    context=ExecutionContext.SANDBOX,
                    tool_name=tool_name,
                    reason="Strict sandbox enforcement",
                )

        return ExecutionDecision(
            context=ExecutionContext.HOST,
            tool_name=tool_name,
            reason="Tool not designated for sandbox",
        )

    def execute_in_sandbox(self, tool_name: str, func: tp.Callable, arguments: dict) -> tp.Any:
        """Execute in sandbox.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            tool_name (str): IN: tool name. OUT: Consumed during execution.
            func (tp.Callable): IN: func. OUT: Consumed during execution.
            arguments (dict): IN: arguments. OUT: Consumed during execution.
        Returns:
            tp.Any: OUT: Result of the operation."""

        if self.backend is None:
            raise SandboxExecutionUnavailableError(tool_name)
        return self.backend.execute(tool_name, func, arguments)
