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


"""Docker-based sandbox backend for Xerxes.

Executes tool functions inside ephemeral Docker containers via the
``docker`` CLI.  The function and its arguments are serialised with
:mod:`pickle`, written to a temporary file that is bind-mounted into the
container, executed by a small Python wrapper, and the result is
serialised as JSON before being returned to the host. Returning pickle
across the container boundary would be an arbitrary-code-execution vector
(see ``pickle.loads`` warnings in the stdlib docs).

Resource limits (timeout, memory, network) from :class:`SandboxConfig`
are translated to ``docker run`` flags.
"""

from __future__ import annotations

import base64
import json
import logging
import pickle
import subprocess
import typing as tp

from ..sandbox import SandboxConfig

logger = logging.getLogger(__name__)


_CONTAINER_RUNNER = """\
import base64, json, pickle, sys

payload = base64.b64decode(sys.stdin.read())
func, args = pickle.loads(payload)
try:
    result = func(**args)
    out = json.dumps({"ok": True, "value": result}, default=repr)
except Exception as exc:
    out = json.dumps({"ok": False, "error": str(exc), "type": type(exc).__name__})
sys.stdout.write(base64.b64encode(out.encode("utf-8")).decode())
"""


class DockerSandboxBackend:
    """Sandbox backend that runs tools inside ephemeral Docker containers.

    Uses the ``docker`` CLI (not the Docker SDK) so there is no extra
    Python dependency.  Requires Docker to be installed and the daemon
    to be running on the host.

    The execution flow is:

    1. The callable and its arguments are serialised with :mod:`pickle`.
    2. The serialised payload is base64-encoded and piped as stdin into a
       ``docker run`` command.
    3. A small Python runner script inside the container deserialises the
       payload, executes the function, and writes the result (or error)
       back to stdout as base64-encoded pickle.
    4. The host deserialises the result and returns it to the caller.

    Resource limits (timeout, memory, network) from :class:`SandboxConfig`
    are translated to ``docker run`` flags.

    Attributes:
        _config: The :class:`SandboxConfig` governing timeout, memory
            limits, network access, and working directory.
        _backend_config: The :class:`SandboxBackendConfig` with
            Docker-specific settings such as image name, mount paths,
            and environment variables.
    """

    def __init__(self, sandbox_config: SandboxConfig) -> None:
        """Initialise the Docker sandbox backend.

        Args:
            sandbox_config: The sandbox configuration containing resource
                limits and backend-specific settings.
        """
        self._config = sandbox_config
        self._backend_config = sandbox_config.backend_config

    def execute(self, tool_name: str, func: tp.Callable, arguments: dict) -> tp.Any:
        """Execute a callable with arguments inside an ephemeral Docker container.

        The function and its arguments are serialised with :mod:`pickle`,
        base64-encoded, and piped into a minimal Python runner script
        inside the container. The result is deserialised from the
        container's stdout.

        Args:
            tool_name: The name of the tool being executed, used for
                logging and error messages.
            func: The callable to execute within the Docker container.
                Must be picklable.
            arguments: Keyword arguments to pass to *func*. All values
                must be picklable.

        Returns:
            The return value of ``func(**arguments)`` as produced inside
            the container.

        Raises:
            RuntimeError: If the Docker command times out, the container
                returns a non-zero exit code, the result cannot be
                deserialised, or the function raised an exception inside
                the container.
        """
        payload = pickle.dumps((func, arguments))
        encoded_payload = base64.b64encode(payload).decode()

        cmd = self._build_docker_command(tool_name)
        logger.debug("Docker sandbox executing tool %r: %s", tool_name, " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd,
                input=encoded_payload,
                capture_output=True,
                text=True,
                timeout=self._config.sandbox_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Docker sandbox execution of tool {tool_name!r} timed out after {self._config.sandbox_timeout}s"
            ) from exc

        if proc.returncode != 0:
            raise RuntimeError(
                f"Docker sandbox execution of tool {tool_name!r} failed (exit {proc.returncode}): {proc.stderr.strip()}"
            )

        try:
            result_bytes = base64.b64decode(proc.stdout)
            result_data: dict = json.loads(result_bytes.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Failed to deserialise sandbox result for tool {tool_name!r}: {exc}") from exc

        if not result_data.get("ok"):
            raise RuntimeError(
                f"Tool {tool_name!r} raised {result_data.get('type', 'Exception')} "
                f"inside sandbox: {result_data.get('error', 'unknown error')}"
            )
        return result_data["value"]

    def is_available(self) -> bool:
        """Check whether Docker is installed and the daemon is running.

        Runs ``docker info`` with a 10-second timeout to verify that the
        ``docker`` CLI is on the PATH and the daemon is responsive.

        Returns:
            ``True`` if the ``docker`` CLI is accessible and the daemon
            responded successfully, ``False`` otherwise (including when
            Docker is not installed, the daemon is stopped, or the
            command times out).
        """
        try:
            proc = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
            )
            return proc.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    def get_capabilities(self) -> dict[str, tp.Any]:
        """Return a dictionary describing the Docker backend's capabilities and status.

        Returns:
            A dict with the following keys:

            - ``"backend"``: Always ``"docker"``.
            - ``"available"``: Whether Docker is currently available.
            - ``"image"``: The container image used for execution.
            - ``"network_access"``: Whether sandbox has network access.
            - ``"memory_limit_mb"``: Memory limit in megabytes.
            - ``"timeout"``: Execution timeout in seconds.
        """
        available = self.is_available()
        return {
            "backend": "docker",
            "available": available,
            "image": self._backend_config.image,
            "network_access": self._config.sandbox_network_access,
            "memory_limit_mb": self._config.sandbox_memory_limit_mb,
            "timeout": self._config.sandbox_timeout,
        }

    def _build_docker_command(self, tool_name: str) -> list[str]:
        """Build the full ``docker run`` command-line argument list.

        Constructs the command with appropriate flags for memory limits,
        network isolation, volume mounts, environment variables, and the
        container image. The container is run with ``--rm`` (auto-cleanup)
        and ``-i`` (interactive stdin).

        Args:
            tool_name: The name of the tool being executed (currently
                unused in command construction but available for future
                per-tool customisation).

        Returns:
            A list of strings suitable for passing to
            :func:`subprocess.run`.
        """
        cmd: list[str] = ["docker", "run", "--rm", "-i"]

        cmd.extend(["--memory", f"{self._config.sandbox_memory_limit_mb}m"])

        if not self._config.sandbox_network_access:
            cmd.extend(["--network", "none"])

        workdir = self._config.working_directory
        if workdir:
            readonly = ":ro" if self._backend_config.mount_readonly else ""
            cmd.extend(["-v", f"{workdir}:/workspace{readonly}"])
            cmd.extend(["-w", "/workspace"])

        for host_path, container_path in self._backend_config.mount_paths.items():
            readonly = ":ro" if self._backend_config.mount_readonly else ""
            cmd.extend(["-v", f"{host_path}:{container_path}{readonly}"])

        for key, value in self._backend_config.env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        cmd.append(self._backend_config.image)
        cmd.extend(["python", "-c", _CONTAINER_RUNNER])

        return cmd
