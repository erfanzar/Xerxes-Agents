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

"""Subprocess-based sandbox backend for Calute.

Provides lightweight isolation by running tool functions in a separate
Python subprocess.  On Unix platforms, :mod:`resource` limits are applied
to the child process for memory capping.

This backend is always available (no Docker required) and is suitable
when *some* process-level isolation is acceptable, although it does
**not** provide filesystem or network sandboxing.
"""

from __future__ import annotations

import base64
import logging
import os
import pickle
import subprocess
import sys
import typing as tp

from ..sandbox import SandboxConfig

logger = logging.getLogger(__name__)

_CHILD_SCRIPT = """\
import base64, os, pickle, sys

# Apply memory limit if provided via environment variable.
mem_limit = os.environ.get("_CALUTE_MEM_LIMIT_BYTES")
if mem_limit:
    try:
        import resource
        limit = int(mem_limit)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ImportError, ValueError, OSError):
        pass  # resource module unavailable (Windows) or limit not settable

payload = base64.b64decode(sys.stdin.read())
func, args = pickle.loads(payload)
try:
    result = func(**args)
    out = pickle.dumps({"ok": True, "value": result})
except Exception as exc:
    out = pickle.dumps({"ok": False, "error": str(exc), "type": type(exc).__name__})
sys.stdout.write(base64.b64encode(out).decode())
"""


class SubprocessSandboxBackend:
    """Sandbox backend that runs tools in a child Python subprocess.

    This provides process-level isolation: a crash or memory overflow in
    the child will not bring down the host process.  It does **not**
    provide filesystem or network isolation.

    On Unix platforms, the :mod:`resource` module is used to apply memory
    limits (``RLIMIT_AS``) to the child process. On Windows, the memory
    limit environment variable is set but may not be enforced if the
    :mod:`resource` module is unavailable.

    The execution flow mirrors :class:`DockerSandboxBackend`:

    1. The callable and arguments are pickle-serialised and base64-encoded.
    2. The encoded payload is piped as stdin to a child Python process.
    3. The child deserialises, executes, and writes the result back to
       stdout as base64-encoded pickle.
    4. The host deserialises and returns the result.

    Attributes:
        _config: The :class:`SandboxConfig` governing timeout, memory
            limits, and working directory for the child process.
    """

    def __init__(self, sandbox_config: SandboxConfig) -> None:
        """Initialise the subprocess sandbox backend.

        Args:
            sandbox_config: The sandbox configuration containing resource
                limits and working directory settings.
        """
        self._config = sandbox_config

    def execute(self, tool_name: str, func: tp.Callable, arguments: dict) -> tp.Any:
        """Execute a callable with arguments in a child Python subprocess.

        The function and its arguments are serialised with :mod:`pickle`,
        base64-encoded, and piped to a child Python process that applies
        memory limits, executes the function, and returns the result via
        stdout.

        Args:
            tool_name: The name of the tool being executed, used for
                logging and error messages.
            func: The callable to execute in the child process. Must be
                picklable.
            arguments: Keyword arguments to pass to *func*. All values
                must be picklable.

        Returns:
            The return value of ``func(**arguments)`` as produced in the
            child process.

        Raises:
            RuntimeError: If the subprocess times out, exits with a
                non-zero code, the result cannot be deserialised, or the
                function raised an exception inside the child process.
        """
        payload = pickle.dumps((func, arguments))
        encoded_payload = base64.b64encode(payload).decode()

        env = os.environ.copy()
        mem_bytes = self._config.sandbox_memory_limit_mb * 1024 * 1024
        env["_CALUTE_MEM_LIMIT_BYTES"] = str(mem_bytes)

        cwd = self._config.working_directory

        cmd = [sys.executable, "-c", _CHILD_SCRIPT]
        logger.debug("Subprocess sandbox executing tool %r", tool_name)

        try:
            proc = subprocess.run(
                cmd,
                input=encoded_payload,
                capture_output=True,
                text=True,
                timeout=self._config.sandbox_timeout,
                env=env,
                cwd=cwd,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Subprocess sandbox execution of tool {tool_name!r} timed out after {self._config.sandbox_timeout}s"
            ) from exc

        if proc.returncode != 0:
            raise RuntimeError(
                f"Subprocess sandbox execution of tool {tool_name!r} failed "
                f"(exit {proc.returncode}): {proc.stderr.strip()}"
            )

        try:
            result_bytes = base64.b64decode(proc.stdout)
            result_data: dict = pickle.loads(result_bytes)
        except Exception as exc:
            raise RuntimeError(f"Failed to deserialise subprocess sandbox result for tool {tool_name!r}: {exc}") from exc

        if not result_data.get("ok"):
            raise RuntimeError(
                f"Tool {tool_name!r} raised {result_data.get('type', 'Exception')} "
                f"inside subprocess sandbox: {result_data.get('error', 'unknown error')}"
            )
        return result_data["value"]

    def is_available(self) -> bool:
        """Check whether the subprocess backend is available.

        This backend is always available since it only requires the
        Python interpreter that is already running the host process.

        Returns:
            Always ``True``.
        """
        return True

    def get_capabilities(self) -> dict[str, tp.Any]:
        """Return a dictionary describing the subprocess backend's capabilities.

        Returns:
            A dict with the following keys:

            - ``"backend"``: Always ``"subprocess"``.
            - ``"available"``: Always ``True``.
            - ``"isolation_level"``: Always ``"process"``.
            - ``"filesystem_isolation"``: Always ``False`` (no filesystem
              sandboxing).
            - ``"network_isolation"``: Always ``False`` (no network
              sandboxing).
            - ``"memory_limit_mb"``: Configured memory limit in megabytes.
            - ``"timeout"``: Configured execution timeout in seconds.
        """
        return {
            "backend": "subprocess",
            "available": True,
            "isolation_level": "process",
            "filesystem_isolation": False,
            "network_isolation": False,
            "memory_limit_mb": self._config.sandbox_memory_limit_mb,
            "timeout": self._config.sandbox_timeout,
        }
