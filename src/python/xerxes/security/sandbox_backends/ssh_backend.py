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
"""SSH-backed sandbox: shell out to ``ssh user@host`` for execution.

Uses the local OpenSSH client (no Python dependency). Configuration
via the ``XERXES_SSH_HOST`` env var or an explicit ``ssh_host`` arg.
The shell command is single-shot — long-running interactive sessions
are out of scope; this backend is for "run this on the build box"."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass
class SshBackendConfig:
    """Connection parameters for the SSH-backed sandbox.

    Attributes:
        host: target hostname or IP.
        user: optional SSH user; falls through to ssh_config when None.
        port: optional non-default port.
        identity_file: optional private key path for ``-i``.
        timeout: subprocess wall-clock cap in seconds.
    """

    host: str
    user: str | None = None
    port: int | None = None
    identity_file: str | None = None
    timeout: int = 60


class SshSandboxBackend:
    """Run a single command on a remote host via the system ssh client."""

    name = "ssh"

    def __init__(self, config: SshBackendConfig | None = None) -> None:
        """Bind to an explicit config or resolve one from env on each call."""
        self._config = config

    def _resolve_config(self, **overrides: Any) -> SshBackendConfig:
        """Return the active config, applying per-call ``overrides`` on a copy."""
        if self._config is not None:
            base = self._config
        else:
            host = overrides.pop("ssh_host", None) or os.environ.get("XERXES_SSH_HOST")
            if not host:
                raise RuntimeError("SSH host required; set XERXES_SSH_HOST or pass ssh_host")
            base = SshBackendConfig(host=host)
        for k, v in overrides.items():
            if hasattr(base, k):
                setattr(base, k, v)
        return base

    def execute(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Run ``command`` over SSH. Returns ``{returncode, stdout, stderr}``."""
        cfg = self._resolve_config(**overrides)
        target = f"{cfg.user}@{cfg.host}" if cfg.user else cfg.host
        argv = ["ssh", "-o", "BatchMode=yes"]
        if cfg.port:
            argv.extend(["-p", str(cfg.port)])
        if cfg.identity_file:
            argv.extend(["-i", cfg.identity_file])
        argv.append(target)
        remote_parts = []
        if cwd:
            remote_parts.append(f"cd {shlex.quote(cwd)}")
        if env:
            for k, v in env.items():
                remote_parts.append(f"export {shlex.quote(k)}={shlex.quote(v)}")
        remote_parts.append(command)
        argv.append(" && ".join(remote_parts))
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=cfg.timeout,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "host": cfg.host,
        }


__all__ = ["SshBackendConfig", "SshSandboxBackend"]
