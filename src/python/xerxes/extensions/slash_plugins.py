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
"""Plugin-registered slash commands.

Plugins call ``register_slash`` to expose a ``/customcmd`` that the TUI
slash completer and the bridge's ``_run_slash`` dispatcher can invoke.
Resolution is merged with the built-in registry from
``xerxes.bridge.commands.COMMAND_REGISTRY``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..bridge.commands import COMMAND_REGISTRY, CommandDef, resolve_command

SlashHandler = Callable[..., Any]


@dataclass(frozen=True)
class SlashPlugin:
    """Pairing of a slash ``CommandDef`` with its plugin-supplied handler.

    Attributes:
        command: Slash command definition.
        handler: Callable invoked when the command is dispatched.
    """

    command: CommandDef
    handler: SlashHandler


class SlashPluginRegistry:
    """Thread-safe registry of plugin-contributed slash commands."""

    def __init__(self) -> None:
        """Initialize an empty plugin registry with a backing lock."""
        self._lock = threading.Lock()
        self._plugins: dict[str, SlashPlugin] = {}

    def register(
        self,
        name: str,
        handler: SlashHandler,
        *,
        description: str = "",
        category: str = "tools",
        aliases: tuple[str, ...] = (),
        args_hint: str = "",
        cli_only: bool = False,
        gateway_only: bool = False,
    ) -> SlashPlugin:
        """Register a plugin slash command.

        Args:
            name: Command name, optionally prefixed with ``/``.
            handler: Callable to invoke when the command is dispatched.
            description: Human-readable summary shown in completion menus.
            category: Logical grouping (e.g. ``tools``).
            aliases: Alternative names that also resolve to this command.
            args_hint: Argument format string displayed in help text.
            cli_only: Restrict the command to the CLI surface.
            gateway_only: Restrict the command to channel gateways.

        Returns:
            The newly registered ``SlashPlugin`` record.

        Raises:
            ValueError: ``name`` is empty or collides with a built-in command.
        """
        clean = name.lstrip("/")
        if not clean:
            raise ValueError("slash command name must be non-empty")
        # Collision check vs the built-in registry — plugins can't shadow core commands.
        if resolve_command(clean) is not None:
            raise ValueError(f"slash command {clean!r} already exists in core registry")
        cmd = CommandDef(
            name=clean,
            description=description or f"plugin-registered command /{clean}",
            category=category,
            aliases=aliases,
            args_hint=args_hint,
            cli_only=cli_only,
            gateway_only=gateway_only,
        )
        plugin = SlashPlugin(command=cmd, handler=handler)
        with self._lock:
            self._plugins[clean] = plugin
        return plugin

    def unregister(self, name: str) -> bool:
        """Remove ``name`` from the registry and return whether it existed."""
        clean = name.lstrip("/")
        with self._lock:
            return self._plugins.pop(clean, None) is not None

    def resolve(self, text: str) -> SlashPlugin | None:
        """Look up the plugin command matching ``text`` (name or alias)."""
        clean = text.lstrip("/").split(" ", 1)[0]
        with self._lock:
            plugin = self._plugins.get(clean)
            if plugin is not None:
                return plugin
            # Check aliases.
            for p in self._plugins.values():
                if clean in p.command.aliases:
                    return p
            return None

    def list(self) -> list[SlashPlugin]:
        """Return registered plugins sorted by command name."""
        with self._lock:
            return sorted(self._plugins.values(), key=lambda p: p.command.name)

    def all_commands(self) -> list[CommandDef]:
        """Return built-in plus plugin commands merged into one sorted list."""
        seen = {c.name for c in COMMAND_REGISTRY}
        plugin_cmds = [p.command for p in self.list() if p.command.name not in seen]
        return list(COMMAND_REGISTRY) + sorted(plugin_cmds, key=lambda c: c.name)


_default = SlashPluginRegistry()


def register_slash(name: str, handler: SlashHandler, **kwargs: Any) -> SlashPlugin:
    """Register a slash command on the module-level default registry."""
    return _default.register(name, handler, **kwargs)


def resolve_slash(text: str) -> SlashPlugin | None:
    """Resolve ``text`` against the module-level default registry."""
    return _default.resolve(text)


def registered_slashes() -> list[SlashPlugin]:
    """Return every slash plugin registered on the default registry."""
    return _default.list()


def registry() -> SlashPluginRegistry:
    """Return the module-level default ``SlashPluginRegistry``."""
    return _default


__all__ = [
    "SlashHandler",
    "SlashPlugin",
    "SlashPluginRegistry",
    "register_slash",
    "registered_slashes",
    "registry",
    "resolve_slash",
]
