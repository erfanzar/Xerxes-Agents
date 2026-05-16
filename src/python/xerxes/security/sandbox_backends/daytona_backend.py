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
"""Daytona cloud IDE sandbox backend."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any


@dataclass
class DaytonaBackendConfig:
    """Provisioning knobs for the Daytona-backed sandbox.

    Attributes:
        workspace_image: container image for the ephemeral workspace.
        region: Daytona region used at workspace creation time.
        timeout_seconds: wall-clock cap per ``exec`` call.
    """

    workspace_image: str = "python:3.11"
    region: str = "us-east-1"
    timeout_seconds: int = 60


class DaytonaSandboxBackend:
    """Provision a Daytona workspace per call and exec ``command`` inside."""

    name = "daytona"

    def __init__(self, config: DaytonaBackendConfig | None = None) -> None:
        """Bind to an explicit config or use defaults."""
        self._config = config or DaytonaBackendConfig()

    def execute(
        self, command: str, *, env: dict[str, str] | None = None, cwd: str | None = None, **_: Any
    ) -> dict[str, Any]:
        """Create a workspace, run ``command``, then tear the workspace down.

        Raises ``RuntimeError`` if the optional ``daytona`` SDK is missing."""
        if importlib.util.find_spec("daytona") is None:
            raise RuntimeError("Daytona SDK not installed; install xerxes-agent[backend-daytona].")
        import daytona  # type: ignore

        client = daytona.Client()  # type: ignore[attr-defined]
        workspace = client.workspaces.create(image=self._config.workspace_image, region=self._config.region)
        try:
            response = workspace.exec(
                command=command,
                cwd=cwd,
                env=env or {},
                timeout=self._config.timeout_seconds,
            )
            return {
                "returncode": response.exit_code,
                "stdout": response.stdout,
                "stderr": response.stderr,
                "workspace_id": workspace.id,
            }
        finally:
            workspace.delete()


__all__ = ["DaytonaBackendConfig", "DaytonaSandboxBackend"]
