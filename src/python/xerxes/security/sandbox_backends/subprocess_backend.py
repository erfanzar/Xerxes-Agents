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
"""Subprocess backend module for Xerxes.

Exports:
    - logger
    - SubprocessSandboxBackend"""

from __future__ import annotations

import base64
import json
import logging
import os
import pickle
import subprocess
import sys
import typing as tp

from ..sandbox import SandboxConfig

logger = logging.getLogger(__name__)

_CHILD_SCRIPT = """\
import base64, json, os, pickle, sys

mem_limit = os.environ.get("_XERXES_MEM_LIMIT_BYTES")
if mem_limit:
    try:
        import resource
        limit = int(mem_limit)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ImportError, ValueError, OSError):
        pass

payload = base64.b64decode(sys.stdin.read())
func, args = pickle.loads(payload)
try:
    result = func(**args)
    out = json.dumps({"ok": True, "value": result}, default=repr)
except Exception as exc:
    out = json.dumps({"ok": False, "error": str(exc), "type": type(exc).__name__})
sys.stdout.write(base64.b64encode(out.encode("utf-8")).decode())
"""


class SubprocessSandboxBackend:
    """Subprocess sandbox backend."""

    def __init__(self, sandbox_config: SandboxConfig) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            sandbox_config (SandboxConfig): IN: sandbox config. OUT: Consumed during execution."""

        self._config = sandbox_config

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

        env = os.environ.copy()
        mem_bytes = self._config.sandbox_memory_limit_mb * 1024 * 1024
        env["_XERXES_MEM_LIMIT_BYTES"] = str(mem_bytes)

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
            result_data: dict = json.loads(result_bytes.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Failed to deserialise subprocess sandbox result for tool {tool_name!r}: {exc}") from exc

        if not result_data.get("ok"):
            raise RuntimeError(
                f"Tool {tool_name!r} raised {result_data.get('type', 'Exception')} "
                f"inside subprocess sandbox: {result_data.get('error', 'unknown error')}"
            )
        return result_data["value"]

    def is_available(self) -> bool:
        """Check whether available.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            bool: OUT: Result of the operation."""

        return True

    def get_capabilities(self) -> dict[str, tp.Any]:
        """Retrieve the capabilities.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        return {
            "backend": "subprocess",
            "available": True,
            "isolation_level": "process",
            "filesystem_isolation": False,
            "network_isolation": False,
            "memory_limit_mb": self._config.sandbox_memory_limit_mb,
            "timeout": self._config.sandbox_timeout,
        }
