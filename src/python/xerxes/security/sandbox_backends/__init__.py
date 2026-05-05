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
"""Init module for Xerxes.

Exports:
    - register_backend
    - get_backend
    - list_backends"""

from __future__ import annotations

import typing as tp

from ..sandbox import SandboxBackend, SandboxBackendConfig, SandboxConfig

if tp.TYPE_CHECKING:
    pass

_BACKEND_REGISTRY: dict[str, type] = {}


def register_backend(name: str, cls: type) -> None:
    """Register backend.

    Args:
        name (str): IN: name. OUT: Consumed during execution.
        cls: IN: The class. OUT: Used for class-level operations."""

    _BACKEND_REGISTRY[name] = cls
    """Register backend.

    Args:
        name (str): IN: name. OUT: Consumed during execution.
        cls: IN: The class. OUT: Used for class-level operations."""
    """Register backend.

    Args:
        name (str): IN: name. OUT: Consumed during execution.
        cls: IN: The class. OUT: Used for class-level operations."""


def get_backend(name: str, sandbox_config: SandboxConfig) -> SandboxBackend:
    """Retrieve the backend.

    Args:
        name (str): IN: name. OUT: Consumed during execution.
        sandbox_config (SandboxConfig): IN: sandbox config. OUT: Consumed during execution.
    Returns:
        SandboxBackend: OUT: Result of the operation."""

    cls = _BACKEND_REGISTRY.get(name)
    """Retrieve the backend.

    Args:
        name (str): IN: name. OUT: Consumed during execution.
        sandbox_config (SandboxConfig): IN: sandbox config. OUT: Consumed during execution.
    Returns:
        SandboxBackend: OUT: Result of the operation."""
    """Retrieve the backend.

    Args:
        name (str): IN: name. OUT: Consumed during execution.
        sandbox_config (SandboxConfig): IN: sandbox config. OUT: Consumed during execution.
    Returns:
        SandboxBackend: OUT: Result of the operation."""
    if cls is None:
        available = ", ".join(sorted(_BACKEND_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown sandbox backend {name!r}. Available backends: {available}")
    return cls(sandbox_config=sandbox_config)


def list_backends() -> list[str]:
    """List backends.

    Returns:
        list[str]: OUT: Result of the operation."""

    return sorted(_BACKEND_REGISTRY)
    """List backends.

    Returns:
        list[str]: OUT: Result of the operation."""
    """List backends.

    Returns:
        list[str]: OUT: Result of the operation."""


def _register_builtins() -> None:
    """Internal helper to register builtins."""

    from .docker_backend import DockerSandboxBackend

    """Internal helper to register builtins.
    """
    """Internal helper to register builtins.
    """
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
