# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.


"""Sandbox and elevated execution configuration for Xerxes.

Provides a configurable abstraction for deciding whether tool execution
runs in the host environment or in a sandboxed context.

Sandbox modes:
    - ``off``: All execution on host (default, simplest).
    - ``warn``: Log warnings for tools that *would* be sandboxed.
    - ``strict``: Require sandbox for designated tools (raises if unavailable).

Elevated execution:
    Some tools need to break out of the sandbox (e.g., file I/O that must
    access the real filesystem). The ``elevated`` flag on a tool marks it
    as exempt from sandboxing.

This module provides the config models, runtime decision layer, and the
:class:`SandboxBackend` protocol that concrete backends must implement.
"""

from __future__ import annotations

import logging
import typing as tp
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SandboxMode(Enum):
    """Sandbox enforcement modes controlling how tool execution is handled.

    Attributes:
        OFF: No sandboxing; all tools execute directly on the host. This is
            the default and simplest mode.
        WARN: Advisory mode; tools that would be sandboxed are logged with a
            warning but still execute on the host.
        STRICT: Enforcement mode; tools designated for sandboxing must execute
            in a sandbox backend. Raises
            :class:`SandboxExecutionUnavailableError` if no backend is
            configured.

    Example:
        >>> mode = SandboxMode.STRICT
        >>> mode.value
        'strict'
    """

    OFF = "off"
    WARN = "warn"
    STRICT = "strict"


@dataclass
class SandboxBackendConfig:
    """Backend-specific settings for sandbox execution.

    Attributes:
        image: Container image name (for Docker backend).
        mount_paths: Host paths to mount into the sandbox (maps host -> container).
        mount_readonly: Whether mounts are read-only by default.
        env_vars: Environment variables to inject into the sandbox.
        extra_args: Additional backend-specific arguments.
    """

    image: str = "python:3.12-slim"
    mount_paths: dict[str, str] = field(default_factory=dict)
    mount_readonly: bool = True
    env_vars: dict[str, str] = field(default_factory=dict)
    extra_args: dict[str, tp.Any] = field(default_factory=dict)


@dataclass
class SandboxConfig:
    """Configuration for sandbox behavior.

    Attributes:
        mode: The sandbox enforcement mode.
        sandboxed_tools: Tools that should run in sandbox when mode != off.
        elevated_tools: Tools exempt from sandbox (run on host always).
        sandbox_timeout: Timeout for sandboxed execution in seconds.
        sandbox_memory_limit_mb: Memory limit for sandboxed processes.
        sandbox_network_access: Whether sandbox has network access.
        working_directory: Working directory inside the sandbox.
        backend_type: Name of the sandbox backend to use (e.g. ``"docker"``, ``"subprocess"``).
        backend_config: Backend-specific settings.
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
    """Describes the runtime environment where a tool will actually execute.

    Attributes:
        HOST: The tool runs directly in the host Python process with full
            access to the filesystem, network, and system resources.
        SANDBOX: The tool runs inside an isolated sandbox environment
            managed by a :class:`SandboxBackend` implementation.

    Example:
        >>> ctx = ExecutionContext.SANDBOX
        >>> ctx.value
        'sandbox'
    """

    HOST = "host"
    SANDBOX = "sandbox"


@dataclass
class ExecutionDecision:
    """Result of the sandbox router's decision for a specific tool invocation.

    Encapsulates where the tool should run and why that decision was made,
    providing an auditable record for logging or policy review.

    Attributes:
        context: The :class:`ExecutionContext` indicating whether the tool
            should run on the host or in a sandbox.
        tool_name: The name of the tool this decision applies to.
        reason: A human-readable explanation of why this execution context
            was chosen.
    """

    context: ExecutionContext
    tool_name: str
    reason: str


class SandboxExecutionUnavailableError(RuntimeError):
    """Raised when strict sandbox execution is required but no backend is configured.

    This error occurs in :attr:`SandboxMode.STRICT` mode when a tool
    designated for sandboxed execution is invoked but no
    :class:`SandboxBackend` has been provided to the :class:`SandboxRouter`.

    Attributes:
        tool_name: The name of the tool that required sandbox execution.
    """

    def __init__(self, tool_name: str) -> None:
        """Initialise the error with context about the failed sandbox request.

        Args:
            tool_name: The name of the tool that required sandbox execution
                but could not be accommodated.
        """
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' requires sandbox execution, but no sandbox backend is configured")


class SandboxBackend(tp.Protocol):
    """Protocol defining the interface that all sandbox backends must implement.

    Concrete implementations (e.g.,
    :class:`~xerxes.security.sandbox_backends.DockerSandboxBackend`,
    :class:`~xerxes.security.sandbox_backends.SubprocessSandboxBackend`)
    must provide all three methods to be used by the :class:`SandboxRouter`.

    Note:
        All methods are synchronous. The :class:`SandboxRouter` calls
        ``execute_in_sandbox`` from an async context via ``run_in_executor``,
        so backends do not need to be async themselves. If a backend requires
        async operations internally, it should handle those within the sync
        ``execute`` method.
    """

    def execute(self, tool_name: str, func: tp.Callable, arguments: dict) -> tp.Any:
        """Execute a tool function inside the sandboxed runtime.

        This method is called synchronously by the SandboxRouter. The router
        ensures this is invoked in a thread pool executor so backends do not
        block the event loop.

        Args:
            tool_name: The name of the tool being executed, used for logging
                and error messages.
            func: The callable to execute within the sandbox.
            arguments: Keyword arguments to pass to *func*.

        Returns:
            The return value of ``func(**arguments)`` as produced inside
            the sandbox.

        Raises:
            RuntimeError: If the sandbox execution fails for any reason
                (timeout, crash, serialisation error, etc.).
        """

    def is_available(self) -> bool:
        """Check whether the backend is ready to accept execution requests.

        Returns:
            ``True`` if the backend's runtime dependencies are satisfied
            and it can execute tools, ``False`` otherwise.
        """

    def get_capabilities(self) -> dict[str, tp.Any]:
        """Return a dictionary describing the backend's capabilities and status.

        Returns:
            A dict containing at minimum a ``"backend"`` key with the
            backend name and an ``"available"`` boolean. Additional
            backend-specific keys (e.g., ``"image"``,
            ``"isolation_level"``) may be included.
        """


class SandboxRouter:
    """Decides the execution context (host vs. sandbox) for tool calls.

    The router inspects the :class:`SandboxConfig` to determine whether a
    given tool should run on the host or be dispatched to a
    :class:`SandboxBackend`. Elevated tools always run on the host,
    regardless of the sandbox mode.

    Attributes:
        config: The :class:`SandboxConfig` governing sandbox behaviour.
        backend: An optional :class:`SandboxBackend` instance used to
            execute sandboxed tools. Required when ``config.mode`` is
            :attr:`SandboxMode.STRICT`.

    Example:
        >>> config = SandboxConfig(mode=SandboxMode.WARN, sandboxed_tools={"execute_shell"})
        >>> router = SandboxRouter(config)
        >>> decision = router.decide("execute_shell")
        >>> decision.context
        <ExecutionContext.HOST: 'host'>
    """

    def __init__(self, config: SandboxConfig | None = None, backend: SandboxBackend | None = None) -> None:
        """Initialise the sandbox router.

        Args:
            config: The sandbox configuration. Defaults to a
                :class:`SandboxConfig` with :attr:`SandboxMode.OFF` if not
                provided.
            backend: An optional concrete :class:`SandboxBackend`
                implementation. Must be provided if ``config.mode`` is
                :attr:`SandboxMode.STRICT` and sandboxed tools will be
                invoked.
        """
        self.config = config or SandboxConfig()
        self.backend = backend

    def decide(self, tool_name: str) -> ExecutionDecision:
        """Determine where a tool should execute based on the current configuration.

        The decision logic follows this precedence:

        1. If the tool is in ``elevated_tools``, it always runs on the host.
        2. If the sandbox mode is :attr:`SandboxMode.OFF`, the tool runs on
           the host.
        3. If the tool is in ``sandboxed_tools`` and mode is
           :attr:`SandboxMode.WARN`, a warning is logged and it runs on
           the host.
        4. If the tool is in ``sandboxed_tools`` and mode is
           :attr:`SandboxMode.STRICT`, it runs in the sandbox.
        5. Otherwise, the tool runs on the host.

        Args:
            tool_name: The name of the tool to route.

        Returns:
            An :class:`ExecutionDecision` containing the chosen
            :class:`ExecutionContext` and the reason for the decision.
        """
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
        """Execute a tool function within the configured sandbox backend.

        This method delegates execution to the :class:`SandboxBackend`
        assigned to this router. It intentionally does **not** fall back to
        host execution; if no backend is configured, an error is raised so
        that strict mode guarantees are upheld.

        Args:
            tool_name: The name of the tool being executed, used for
                logging and error reporting.
            func: The callable to execute within the sandbox.
            arguments: Keyword arguments to pass to *func*.

        Returns:
            The return value of ``func(**arguments)`` as produced inside
            the sandbox.

        Raises:
            SandboxExecutionUnavailableError: If no :class:`SandboxBackend`
                is configured on this router.
            RuntimeError: If the sandbox backend encounters an execution
                error (timeout, crash, serialisation failure, etc.).
        """
        if self.backend is None:
            raise SandboxExecutionUnavailableError(tool_name)
        return self.backend.execute(tool_name, func, arguments)
