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
"""Docker backend module for Xerxes.

Exports:
    - logger
    - DockerSandboxBackend"""

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
    """Docker sandbox backend."""

    def __init__(self, sandbox_config: SandboxConfig) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            sandbox_config (SandboxConfig): IN: sandbox config. OUT: Consumed during execution."""

        self._config = sandbox_config
        self._backend_config = sandbox_config.backend_config

    def execute(self, tool_name: str, func: tp.Callable, arguments: dict) -> tp.Any:
        """Execute.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            tool_name (str): IN: tool name. OUT: Consumed during execution.
            func (tp.Callable): IN: func. OUT: Consumed during execution.
            arguments (dict): IN: arguments. OUT: Consumed during execution.
        Returns:
            tp.Any: OUT: Result of the operation."""

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
        """Check whether available.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            bool: OUT: Result of the operation."""

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
        """Retrieve the capabilities.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
        """Internal helper to build docker command.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            tool_name (str): IN: tool name. OUT: Consumed during execution.
        Returns:
            list[str]: OUT: Result of the operation."""

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
