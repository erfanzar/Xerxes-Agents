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
"""Singularity / Apptainer container sandbox backend.

For HPC environments where Docker isn't permitted. Uses the
``singularity exec`` or ``apptainer exec`` binary if present."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass
class SingularityBackendConfig:
    """Configuration for the Singularity/Apptainer backend.

    Attributes:
        image: image URI; ``docker://`` URIs are pulled and converted on use.
        timeout_seconds: wall-clock cap per execution.
    """

    image: str = "docker://python:3.11-slim"
    timeout_seconds: int = 60


class SingularitySandboxBackend:
    """Shell out to ``singularity exec`` / ``apptainer exec`` for HPC sites."""

    name = "singularity"

    def __init__(self, config: SingularityBackendConfig | None = None) -> None:
        """Bind the backend to an optional explicit configuration."""
        self._config = config or SingularityBackendConfig()

    @staticmethod
    def _resolve_binary() -> str:
        """Locate the singularity or apptainer binary on PATH or raise."""
        for candidate in ("singularity", "apptainer"):
            path = shutil.which(candidate)
            if path is not None:
                return path
        raise RuntimeError("Neither singularity nor apptainer found on PATH")

    def execute(
        self, command: str, *, env: dict[str, str] | None = None, cwd: str | None = None, **_: Any
    ) -> dict[str, Any]:
        """Run ``command`` inside the container and return its output dict.

        The result has the same shape as the other shell-style backends:
        ``{returncode, stdout, stderr, image}``."""
        binary = self._resolve_binary()
        argv = [binary, "exec"]
        if cwd:
            argv.extend(["--pwd", cwd])
        argv.append(self._config.image)
        argv.extend(["bash", "-lc", command])
        envvars = {**env} if env else None
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=self._config.timeout_seconds,
            env=envvars,
        )
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "image": self._config.image}


__all__ = ["SingularityBackendConfig", "SingularitySandboxBackend"]
