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
"""Channel registry and lifecycle management.

Provides ``ChannelRegistry`` for collecting, starting, and stopping multiple
channels, plus ``gather_inbound`` for starting several registries concurrently.
"""

from __future__ import annotations

import asyncio
import logging
import typing as tp

from .base import Channel
from .types import ChannelMessage

logger = logging.getLogger(__name__)
InboundHandler = tp.Callable[[ChannelMessage], tp.Awaitable[None]]


class ChannelRegistry:
    """Holds a collection of named channels and manages their lifecycle."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._channels: dict[str, Channel] = {}
        self._handler: InboundHandler | None = None
        self._started: set[str] = set()

    def register(self, name: str, channel: Channel) -> None:
        """Add a channel to the registry.

        Args:
            name (str): IN: unique identifier for the channel.
            channel (Channel): IN: channel instance to register.
        """
        self._channels[name] = channel

    def unregister(self, name: str) -> None:
        """Remove a channel from the registry.

        Args:
            name (str): IN: identifier of the channel to remove.
        """
        self._channels.pop(name, None)
        self._started.discard(name)

    def get(self, name: str) -> Channel | None:
        """Retrieve a channel by name.

        Args:
            name (str): IN: channel identifier.

        Returns:
            Channel | None: OUT: the channel if registered.
        """
        return self._channels.get(name)

    def all(self) -> dict[str, Channel]:
        """Return all registered channels.

        Returns:
            dict[str, Channel]: OUT: shallow copy of the internal mapping.
        """
        return dict(self._channels)

    def names(self) -> list[str]:
        """Return the names of all registered channels.

        Returns:
            list[str]: OUT: list of registered channel names.
        """
        return list(self._channels.keys())

    def set_handler(self, handler: InboundHandler) -> None:
        """Set the global inbound handler for all channels.

        Args:
            handler (InboundHandler): IN: async callback invoked for each
                inbound message.
        """
        self._handler = handler

    async def start_all(self) -> None:
        """Start all registered channels that are not already running.

        Raises:
            RuntimeError: If ``set_handler`` has not been called.
        """
        if self._handler is None:
            raise RuntimeError("ChannelRegistry.set_handler must be called before start_all()")
        for name, channel in list(self._channels.items()):
            if name in self._started:
                continue
            try:
                await channel.start(self._handler)
                self._started.add(name)
                logger.info("Channel started: %s", name)
            except Exception:
                logger.warning("Failed to start channel %s", name, exc_info=True)

    async def stop_all(self) -> None:
        """Stop all currently running channels."""
        for name in list(self._started):
            channel = self._channels.get(name)
            if channel is None:
                self._started.discard(name)
                continue
            try:
                await channel.stop()
            except Exception:
                logger.debug("Channel %s raised on stop", name, exc_info=True)
            self._started.discard(name)

    async def send(self, message: ChannelMessage) -> None:
        """Send a message through its designated channel.

        Args:
            message (ChannelMessage): IN: message whose ``channel`` field
                determines the target.

        Raises:
            KeyError: If the target channel is not registered.
        """
        chan = self._channels.get(message.channel)
        if chan is None:
            raise KeyError(f"unknown channel {message.channel!r}")
        await chan.send(message)

    def discover_entry_points(self, group: str = "xerxes.channels") -> list[str]:
        """Discover and auto-register channels from package entry points.

        Args:
            group (str): IN: entry-point group to scan. Defaults to
                ``"xerxes.channels"``.

        Returns:
            list[str]: OUT: names of channels that were successfully loaded
            and registered.
        """
        try:
            from importlib.metadata import entry_points
        except Exception:
            return []
        added: list[str] = []
        try:
            eps = entry_points(group=group)
        except TypeError:
            eps = entry_points().get(group, tp.cast(tp.Any, []))
        for ep in eps:
            try:
                factory = ep.load()
                chan = factory()
                if isinstance(chan, Channel):
                    self.register(ep.name, chan)
                    added.append(ep.name)
            except Exception:
                logger.warning("Channel entry-point %s failed to load", ep.name, exc_info=True)
        return added


def gather_inbound(*registries: ChannelRegistry) -> tp.Awaitable[None]:
    """Start all supplied registries concurrently.

    Args:
        *registries (ChannelRegistry): IN: one or more registries to start.

    Returns:
        Awaitable[None]: OUT: coroutine that awaits ``start_all`` on every
        registry.
    """

    async def _run():
        """Asynchronously Internal helper to run.

        Returns:
            Any: OUT: Result of the operation."""
        await asyncio.gather(*(r.start_all() for r in registries))
        """Asynchronously Internal helper to run.

        Returns:
            Any: OUT: Result of the operation."""
        """Asynchronously Internal helper to run.

        Returns:
            Any: OUT: Result of the operation."""

    return _run()


__all__ = ["ChannelRegistry", "InboundHandler", "gather_inbound"]
