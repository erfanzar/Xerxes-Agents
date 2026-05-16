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
"""Modal serverless sandbox backend.

Spawns a Modal sandbox container, runs the command, captures
``stdout/stderr/returncode``. Heavy work — the ``modal`` package is
lazily imported and only when execute() is invoked, so importing
this module is cheap even on installs that don't have the extra."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModalBackendConfig:
    """Modal sandbox provisioning knobs.

    Attributes:
        image: container image used for the sandbox.
        cpu: vCPU allocation passed to ``Sandbox.create``.
        memory_mb: memory cap in MiB.
        timeout_seconds: wall-clock cap.
        env: additional env vars forwarded through ``Secret.from_dict``.
    """

    image: str = "python:3.11-slim"
    cpu: float = 1.0
    memory_mb: int = 1024
    timeout_seconds: int = 60
    env: dict[str, str] = field(default_factory=dict)


class ModalSandboxBackend:
    """Modal-backed sandbox executor."""

    name = "modal"

    def __init__(self, config: ModalBackendConfig | None = None) -> None:
        """Store an explicit config or use a defaulted one on first execute."""
        self._config = config or ModalBackendConfig()

    def _check_modal(self) -> Any:
        """Import the optional ``modal`` SDK; raise with install hint if absent."""
        if importlib.util.find_spec("modal") is None:
            raise RuntimeError(
                "Modal SDK not installed; install xerxes-agent[backend-modal] and run `modal token new` to authenticate."
            )
        import modal  # type: ignore

        return modal

    def execute(
        self, command: str, *, env: dict[str, str] | None = None, cwd: str | None = None, **_: Any
    ) -> dict[str, Any]:
        """Run ``command`` inside a Modal sandbox.

        The actual invocation uses Modal's ``Sandbox`` API; the call
        chain matches Modal v1's Python SDK semantics. When the SDK is
        unavailable, ``RuntimeError`` surfaces immediately."""
        modal = self._check_modal()
        # Build the image dynamically so users can override per-call.
        image = modal.Image.from_registry(self._config.image)  # type: ignore[attr-defined]
        app = modal.App.lookup("xerxes-sandbox", create_if_missing=True)  # type: ignore[attr-defined]
        full_env = {**self._config.env, **(env or {})}
        # Prepend cd so the working dir matches the caller's expectation.
        run_cmd = ["bash", "-lc", (f"cd {cwd} && " if cwd else "") + command]
        with modal.Sandbox.create(  # type: ignore[attr-defined]
            *run_cmd,
            image=image,
            cpu=self._config.cpu,
            memory=self._config.memory_mb,
            timeout=self._config.timeout_seconds,
            app=app,
            secrets=[modal.Secret.from_dict(full_env)] if full_env else None,  # type: ignore[attr-defined]
        ) as sandbox:
            sandbox.wait()
            return {
                "returncode": sandbox.returncode,
                "stdout": sandbox.stdout.read(),
                "stderr": sandbox.stderr.read(),
                "sandbox_id": getattr(sandbox, "object_id", ""),
            }


__all__ = ["ModalBackendConfig", "ModalSandboxBackend"]
