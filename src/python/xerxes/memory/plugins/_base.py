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
"""Shared helpers for external memory plugin scaffolding.

The eight built-in plugins (honcho, mem0, hindsight, holographic,
retaindb, openviking, supermemory, byterover) share a common
shape — they expose 4 standard tools (``add``, ``list``, ``search``,
``remove``) and dispatch to whichever upstream SDK is configured.

These adapters intentionally do NOT depend on the upstream SDK at
import time. ``is_available`` checks env + ``importlib.util.find_spec``;
``handle_tool_call`` lazily imports and raises a clear error when
the SDK is missing. Tests can subclass and override ``_call_upstream``
to avoid network."""

from __future__ import annotations

import importlib.util
import os
import time
from collections.abc import Callable
from typing import Any

from ..provider import MemoryProvider, MemoryToolCall


def _now() -> float:
    """Return the current Unix timestamp as a float."""
    return time.time()


class ExternalMemoryProviderBase(MemoryProvider):
    """Common scaffolding for HTTP- or SDK-backed memory providers.

    Subclasses set the following class attributes:

    * ``name``: plugin name (must match the file/directory).
    * ``required_env``: env vars needed for availability.
    * ``required_module``: optional Python module that must import.
    * ``namespace_label``: short tool-namespace prefix
      (e.g. ``"honcho"`` -> tools become ``honcho_add`` etc.).

    Subclasses override ``_call_upstream`` to perform the actual
    backend interaction. The base class implements the four standard
    tools (``add``/``search``/``list``/``remove``), availability checks
    against env+module, and a no-op lifecycle that flips
    ``_initialized``."""

    required_env: tuple[str, ...] = ()
    required_module: str | None = None
    namespace_label: str = ""

    def __init__(self) -> None:
        """Mark the provider as not yet initialised."""
        self._initialized = False

    # ----- availability + lifecycle ----------------------------------------

    def is_available(self) -> bool:
        """Return True when every required env var is set and module importable."""
        if self.required_module is not None and importlib.util.find_spec(self.required_module) is None:
            return False
        for var in self.required_env:
            if not os.environ.get(var):
                return False
        return True

    def initialize(self) -> None:
        """Flip the initialisation flag; subclasses extend with backend setup."""
        self._initialized = True

    def shutdown(self) -> None:
        """Clear the initialisation flag; subclasses extend with backend teardown."""
        self._initialized = False

    # ----- tool surface -----------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return the four standard tool schemas (add/search/list/remove) prefixed by ``namespace_label``."""
        label = self.namespace_label or self.name
        return [
            {
                "name": f"{label}_add",
                "description": f"Add a memory entry to the {label} backend.",
                "input_schema": {
                    "type": "object",
                    "required": ["content"],
                    "properties": {
                        "content": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            {
                "name": f"{label}_search",
                "description": f"Search {label} memory for relevant entries.",
                "input_schema": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                },
            },
            {
                "name": f"{label}_list",
                "description": f"List recent {label} memory entries.",
                "input_schema": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer"}},
                },
            },
            {
                "name": f"{label}_remove",
                "description": f"Remove a {label} memory entry by id.",
                "input_schema": {
                    "type": "object",
                    "required": ["entry_id"],
                    "properties": {"entry_id": {"type": "string"}},
                },
            },
        ]

    def handle_tool_call(self, call: MemoryToolCall) -> dict[str, Any]:
        """Dispatch a tool call to ``_call_upstream`` and wrap success/error in a JSON-safe dict.

        Returns ``{"ok": False, ...}`` when the backend is unavailable
        or the action name is unknown. Auto-initialises the provider on
        first call."""
        if not self.is_available():
            return {"ok": False, "error": f"{self.name} backend not available"}
        if not self._initialized:
            self.initialize()
        label = self.namespace_label or self.name
        action = call.name.removeprefix(f"{label}_") if call.name.startswith(f"{label}_") else call.name
        if action not in {"add", "search", "list", "remove"}:
            return {"ok": False, "error": f"unknown action: {call.name}"}
        try:
            payload = self._call_upstream(action, call.arguments)
            return {"ok": True, "action": action, "result": payload, "ts": _now()}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "action": action}

    # ----- subclass hook ----------------------------------------------------

    def _call_upstream(
        self, action: str, arguments: dict[str, Any]
    ) -> Any:  # pragma: no cover — overridden by subclasses
        """Execute one backend action; concrete plugins must override.

        Raises:
            NotImplementedError: Always — subclass must implement."""
        raise NotImplementedError(f"{type(self).__name__} must implement _call_upstream")


def make_simple_provider(
    plugin_name: str,
    *,
    required_module: str | None = None,
    required_env: tuple[str, ...] = (),
    upstream_caller: Callable[[str, dict[str, Any]], Any] | None = None,
) -> ExternalMemoryProviderBase:
    """Build an in-process ``ExternalMemoryProviderBase`` for tests.

    When ``upstream_caller`` is omitted, the provider keeps its state
    in a local dict so the plugin still services add/search/list/remove
    calls without hitting any network. Useful for exercising the
    plugin-registry plumbing in CI."""

    state: dict[str, dict[str, Any]] = {}

    def default_caller(action: str, args: dict[str, Any]) -> Any:
        """Default in-memory implementation of the four standard plugin actions."""
        if action == "add":
            entry_id = f"mem_{len(state) + 1:04d}"
            state[entry_id] = {"id": entry_id, "content": args.get("content", ""), "tags": list(args.get("tags", []))}
            return state[entry_id]
        if action == "list":
            limit = int(args.get("limit", 20) or 20)
            return list(state.values())[-limit:]
        if action == "search":
            q = (args.get("query") or "").lower()
            return [e for e in state.values() if q in e["content"].lower()]
        if action == "remove":
            return {"removed": bool(state.pop(args.get("entry_id", ""), None))}
        raise ValueError(f"unknown action: {action}")

    class _Stub(ExternalMemoryProviderBase):
        name = plugin_name
        namespace_label = plugin_name

    _Stub.required_module = required_module
    _Stub.required_env = required_env
    stub = _Stub()
    fn = upstream_caller or default_caller
    stub._call_upstream = fn  # type: ignore[assignment]
    return stub


__all__ = ["ExternalMemoryProviderBase", "make_simple_provider"]
