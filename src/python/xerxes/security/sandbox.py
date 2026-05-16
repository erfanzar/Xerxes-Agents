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
"""Sandbox configuration and routing.

Defines the data model and the host/sandbox routing decision logic used
by the tool dispatcher. The actual isolation mechanism is supplied by a
``SandboxBackend`` implementation (Docker, subprocess, Modal, SSH,
Singularity, Daytona). Threat model: protect the host from tool calls
the model may make on its own initiative (file writes, shell commands,
Python eval) by running designated tools inside an isolated environment
with controlled mounts, env, network, memory and timeout."""

from __future__ import annotations

import logging
import typing as tp
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SandboxMode(Enum):
    """Enforcement level for the sandbox router.

    Values:
        OFF: sandboxing disabled; every tool runs on the host.
        WARN: log a warning when a sandboxed tool runs, but still run on
            host. Useful for dry-runs and migration.
        STRICT: refuse to run a sandboxed tool on the host — route it to
            the configured backend or raise.
    """

    OFF = "off"
    WARN = "warn"
    STRICT = "strict"


@dataclass
class SandboxBackendConfig:
    """Backend-specific knobs (image, mounts, env).

    Attributes:
        image: container image identifier (interpreted by the backend).
        mount_paths: ``{host_path: container_path}`` bind mounts.
        mount_readonly: whether mounts are read-only.
        env_vars: environment variables forwarded into the sandbox.
        extra_args: backend-specific overrides; unrecognised keys are ignored.
    """

    image: str = "python:3.12-slim"
    mount_paths: dict[str, str] = field(default_factory=dict)
    mount_readonly: bool = True
    env_vars: dict[str, str] = field(default_factory=dict)
    extra_args: dict[str, tp.Any] = field(default_factory=dict)


@dataclass
class SandboxConfig:
    """Sandbox policy + resource limits applied across all backends.

    Attributes:
        mode: overall enforcement level (see :class:`SandboxMode`).
        sandboxed_tools: tools that must run inside the sandbox under STRICT.
        elevated_tools: tools explicitly allowed to bypass the sandbox.
        sandbox_timeout: per-call wall-clock limit in seconds.
        sandbox_memory_limit_mb: per-call memory cap in MiB.
        sandbox_network_access: if False, network is disabled in the sandbox.
        working_directory: host path used as the sandbox CWD/mount.
        backend_type: backend name to instantiate (``docker``, ``subprocess``...).
        backend_config: backend-specific configuration.
    """

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
    """Which execution context a tool call landed in."""

    HOST = "host"
    SANDBOX = "sandbox"


@dataclass
class ExecutionDecision:
    """Routing decision for a single tool call.

    Attributes:
        context: where the tool will actually run.
        tool_name: tool the decision applies to.
        reason: short human-readable explanation (mode, allow-list, etc.).
    """

    context: ExecutionContext
    tool_name: str
    reason: str


class SandboxExecutionUnavailableError(RuntimeError):
    """Raised when a sandboxed tool is requested but no backend is wired."""

    def __init__(self, tool_name: str) -> None:
        """Record the offending tool and format a clear error message."""

        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' requires sandbox execution, but no sandbox backend is configured")


class SandboxBackend(tp.Protocol):
    """Structural interface every sandbox backend implementation honours."""

    def execute(self, tool_name: str, func: tp.Callable, arguments: dict) -> tp.Any:
        """Run ``func(**arguments)`` inside the sandbox and return its result."""
        ...

    def is_available(self) -> bool:
        """Return True if the backend's underlying runtime is reachable."""
        ...

    def get_capabilities(self) -> dict[str, tp.Any]:
        """Describe what isolation, limits and image the backend provides."""
        ...


class SandboxRouter:
    """Decide and (optionally) execute tool calls under the active policy.

    The router consults ``SandboxConfig`` to choose a context. Elevated
    tools always run on the host; in OFF mode everything does; in WARN
    mode sandboxed tools log a warning but still execute on host; in
    STRICT mode they are routed to the backend. If STRICT routes a call
    but no backend is attached, :class:`SandboxExecutionUnavailableError`
    is raised."""

    def __init__(self, config: SandboxConfig | None = None, backend: SandboxBackend | None = None) -> None:
        """Bind the router to a configuration and (optionally) a backend."""

        self.config = config or SandboxConfig()
        self.backend = backend

    def decide(self, tool_name: str) -> ExecutionDecision:
        """Return the :class:`ExecutionDecision` for ``tool_name`` under current config."""

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
        """Dispatch ``func(**arguments)`` to the attached backend.

        Raises :class:`SandboxExecutionUnavailableError` if no backend is
        configured. Callers should obtain a HOST decision from :meth:`decide`
        before invoking this method when STRICT mode is not desired."""

        if self.backend is None:
            raise SandboxExecutionUnavailableError(tool_name)
        return self.backend.execute(tool_name, func, arguments)
