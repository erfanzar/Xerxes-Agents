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
"""External memory provider ABC and plugin registry.

Defines the interface that each third-party memory backend (Honcho,
Mem0, Hindsight, Holographic,
RetainDB, OpenViking, Supermemory, ByteRover) implements. Only one
external provider is active at a time; the built-in multi-tier memory
always coexists alongside it.

Exports the abstract ``MemoryProvider`` base, the model-facing
``MemoryToolCall`` value type, a thread-safe ``PluginRegistry`` with a
single-active-provider rule, and the module-level helpers
(``register``, ``active``, ``set_active``, ``registry``) that operate
on the default process-wide registry."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryToolCall:
    """A model-invoked memory tool call.

    Attributes:
        name: Fully-qualified tool name as produced by the model
            (e.g. ``"honcho_search"``).
        arguments: JSON-decoded keyword arguments supplied by the model."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


class MemoryProvider(ABC):
    """Abstract base for every external memory backend.

    Subclasses set a ``name`` class attribute, implement availability /
    initialisation / shutdown lifecycle, expose tool schemas, and
    handle individual tool calls. The optional lifecycle hooks
    (``on_turn_start``, ``on_session_end``, ``on_pre_compress``,
    ``on_memory_write``, ``on_delegation``) default to no-ops so
    backends only implement what they care about."""

    name: str = ""

    @abstractmethod
    def initialize(self) -> None:
        """Perform any one-time setup (open clients, create tables, ...)."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True when this backend can serve requests in the current environment."""
        ...

    @abstractmethod
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return JSON-schema tool definitions for the model to call."""
        ...

    @abstractmethod
    def handle_tool_call(self, call: MemoryToolCall) -> dict[str, Any]:
        """Execute ``call`` against the backend and return a JSON-safe response."""
        ...

    # The following hooks are intentionally optional. Plugins can override
    # any of them, but the base class provides no-op defaults so backends
    # that don't care about every lifecycle event don't need stubs.
    # Each method carries a B027 suppression because ruff's
    # "empty-method-on-ABC" rule assumes we want @abstractmethod here; we
    # explicitly do not.

    def on_turn_start(self, state: Any) -> None:  # noqa: B027
        """Lifecycle hook fired before each turn; default is a no-op."""

    def on_session_end(self, state: Any) -> None:  # noqa: B027
        """Lifecycle hook fired when a session terminates; default is a no-op."""

    def on_pre_compress(self, state: Any) -> None:  # noqa: B027
        """Lifecycle hook fired before context compression runs; default is a no-op."""

    def on_memory_write(self, ref: str, content: str) -> None:  # noqa: B027
        """Lifecycle hook fired after a memory entry is written; default is a no-op."""

    def on_delegation(self, parent_session_id: str, child_session_id: str) -> None:  # noqa: B027
        """Lifecycle hook fired when a subagent is spawned; default is a no-op."""

    def shutdown(self) -> None:  # noqa: B027
        """Release resources held by the provider; default is a no-op."""


class PluginRegistry:
    """Thread-safe registry of memory providers with single-active enforcement.

    All mutating methods take an internal ``threading.Lock`` so the
    registry is safe to access from concurrent runtime threads."""

    def __init__(self) -> None:
        """Initialise the empty provider table and the activation slot."""
        self._providers: dict[str, MemoryProvider] = {}
        self._active: str | None = None
        self._lock = threading.Lock()

    def register(self, provider: MemoryProvider) -> None:
        """Add ``provider`` to the registry under its ``name`` attribute.

        Raises:
            ValueError: ``provider.name`` is empty."""
        if not provider.name:
            raise ValueError(f"memory provider {type(provider).__name__} has no name")
        with self._lock:
            self._providers[provider.name] = provider

    def unregister(self, name: str) -> bool:
        """Remove the named provider and clear the active slot if it pointed there."""
        with self._lock:
            removed = self._providers.pop(name, None) is not None
            if self._active == name:
                self._active = None
            return removed

    def list_names(self) -> list[str]:
        """Return a sorted list of every registered provider name."""
        with self._lock:
            return sorted(self._providers.keys())

    def get(self, name: str) -> MemoryProvider | None:
        """Return the registered provider with ``name`` or ``None``."""
        with self._lock:
            return self._providers.get(name)

    def set_active(self, name: str | None) -> MemoryProvider | None:
        """Switch the active provider to ``name`` (or clear it with ``None``).

        Raises:
            KeyError: ``name`` is not a known provider."""
        with self._lock:
            if name is None:
                self._active = None
                return None
            if name not in self._providers:
                raise KeyError(f"unknown memory provider: {name}")
            self._active = name
            return self._providers[name]

    def active(self) -> MemoryProvider | None:
        """Return the currently active provider or ``None``."""
        with self._lock:
            if self._active is None:
                return None
            return self._providers.get(self._active)


_default_registry = PluginRegistry()


def register(provider: MemoryProvider) -> None:
    """Register ``provider`` on the process-wide default registry."""
    _default_registry.register(provider)


def active() -> MemoryProvider | None:
    """Return the currently active provider on the default registry."""
    return _default_registry.active()


def set_active(name: str | None) -> MemoryProvider | None:
    """Switch the default registry's active provider to ``name``."""
    return _default_registry.set_active(name)


def registry() -> PluginRegistry:
    """Return the process-wide default ``PluginRegistry`` instance."""
    return _default_registry


__all__ = ["MemoryProvider", "MemoryToolCall", "PluginRegistry", "active", "register", "registry", "set_active"]
