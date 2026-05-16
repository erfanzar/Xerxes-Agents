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
"""Registry of sandbox-backend implementations.

Each backend (docker, subprocess, modal, daytona, ssh, singularity)
registers itself here under a short name. ``SandboxRouter`` looks up
the active backend by ``SandboxConfig.backend_type``. Built-in
backends are registered on import; third-party callers can add their
own via :func:`register_backend`."""

from __future__ import annotations

import typing as tp

from ..sandbox import SandboxBackend, SandboxBackendConfig, SandboxConfig

if tp.TYPE_CHECKING:
    pass

_BACKEND_REGISTRY: dict[str, type] = {}


def register_backend(name: str, cls: type) -> None:
    """Add a backend class under ``name``; later calls overwrite earlier ones."""

    _BACKEND_REGISTRY[name] = cls


def get_backend(name: str, sandbox_config: SandboxConfig) -> SandboxBackend:
    """Instantiate the backend registered as ``name`` with the given config.

    Raises ``ValueError`` with the list of registered backends when ``name``
    is unknown."""

    cls = _BACKEND_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_BACKEND_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown sandbox backend {name!r}. Available backends: {available}")
    return cls(sandbox_config=sandbox_config)


def list_backends() -> list[str]:
    """Return all currently registered backend names in sorted order."""

    return sorted(_BACKEND_REGISTRY)


def _register_builtins() -> None:
    """Register the docker + subprocess backends shipped in-tree."""

    from .docker_backend import DockerSandboxBackend
    from .subprocess_backend import SubprocessSandboxBackend

    register_backend("docker", DockerSandboxBackend)
    register_backend("subprocess", SubprocessSandboxBackend)


_register_builtins()

from .docker_backend import DockerSandboxBackend
from .subprocess_backend import SubprocessSandboxBackend

__all__ = [
    "DockerSandboxBackend",
    "SubprocessSandboxBackend",
    "get_backend",
    "list_backends",
    "register_backend",
]
