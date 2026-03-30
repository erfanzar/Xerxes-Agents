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

"""Sandbox backend implementations for Calute.

Provides a registry of concrete :class:`~calute.security.sandbox.SandboxBackend`
implementations and a factory function for instantiation by name.
"""

from __future__ import annotations

import typing as tp

from ..sandbox import SandboxBackend, SandboxBackendConfig, SandboxConfig

if tp.TYPE_CHECKING:
    pass

_BACKEND_REGISTRY: dict[str, type] = {}


def register_backend(name: str, cls: type) -> None:
    """Register a sandbox backend class under the given name in the global registry.

    Once registered, the backend can be instantiated by name via
    :func:`get_backend`. Registering a name that already exists will
    silently overwrite the previous entry.

    Args:
        name: The unique string key for the backend (e.g., ``"docker"``,
            ``"subprocess"``).
        cls: The backend class to register. It must accept a
            ``sandbox_config`` keyword argument of type
            :class:`~calute.security.sandbox.SandboxConfig` in its
            constructor.
    """
    _BACKEND_REGISTRY[name] = cls


def get_backend(name: str, sandbox_config: SandboxConfig) -> SandboxBackend:
    """Instantiate a sandbox backend by name.

    Args:
        name: Registered backend name (e.g. ``"docker"``, ``"subprocess"``).
        sandbox_config: The sandbox configuration to pass to the backend.

    Returns:
        An instance satisfying the :class:`SandboxBackend` protocol.

    Raises:
        ValueError: If *name* is not a registered backend.
    """
    cls = _BACKEND_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_BACKEND_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown sandbox backend {name!r}. Available backends: {available}")
    return cls(sandbox_config=sandbox_config)


def list_backends() -> list[str]:
    """Return the names of all registered sandbox backends, sorted alphabetically.

    Returns:
        A sorted list of backend name strings that have been registered
        via :func:`register_backend` (including the built-in ``"docker"``
        and ``"subprocess"`` backends).

    Example:
        >>> list_backends()
        ['docker', 'subprocess']
    """
    return sorted(_BACKEND_REGISTRY)


def _register_builtins() -> None:
    """Register the built-in Docker and subprocess backends into the global registry.

    This function is called at module import time to ensure that the
    default backends are always available without explicit registration.
    """
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
