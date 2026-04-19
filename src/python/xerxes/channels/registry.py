# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.
"""Channel registry — discovery + lifecycle for messaging adapters.

Higher-level wrapper around :class:`PluginRegistry` that adds:

- Bulk start/stop of all registered channels.
- A per-channel inbound dispatcher that fans :class:`ChannelMessage`
  events from any adapter to a single user-facing handler.
- Optional Python entry-point discovery via the ``xerxes.channels``
  group (``pyproject.toml`` :code:`[project.entry-points."xerxes.channels"]`).
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
    """Manage the lifecycle of all attached :class:`Channel` adapters.

    Intended to be created once per process. Adapters are registered
    by name, started together via :meth:`start_all`, and stopped via
    :meth:`stop_all`.

    Example:
        >>> r = ChannelRegistry()
        >>> r.register("telegram", TelegramChannel(token="..."))
        >>> r.set_handler(my_inbound_handler)
        >>> await r.start_all()
        >>> ...
        >>> await r.stop_all()
    """

    def __init__(self) -> None:
        """Initialise an empty registry with no handler and nothing started."""
        self._channels: dict[str, Channel] = {}
        self._handler: InboundHandler | None = None
        self._started: set[str] = set()

    def register(self, name: str, channel: Channel) -> None:
        """Add a channel under *name* (replaces any existing entry)."""
        self._channels[name] = channel

    def unregister(self, name: str) -> None:
        """Remove the channel ``name`` and forget whether it was started."""
        self._channels.pop(name, None)
        self._started.discard(name)

    def get(self, name: str) -> Channel | None:
        """Return the registered channel ``name`` or ``None`` if missing."""
        return self._channels.get(name)

    def all(self) -> dict[str, Channel]:
        """Return a shallow copy of the name-to-channel mapping."""
        return dict(self._channels)

    def names(self) -> list[str]:
        """Return the list of currently registered channel names."""
        return list(self._channels.keys())

    def set_handler(self, handler: InboundHandler) -> None:
        """Single coroutine that receives every inbound :class:`ChannelMessage`."""
        self._handler = handler

    async def start_all(self) -> None:
        """Start every registered channel, dispatching inbound to the handler."""
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
        """Stop every channel that was started by this registry."""
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
        """Route an outbound message to the channel named in ``message.channel``."""
        chan = self._channels.get(message.channel)
        if chan is None:
            raise KeyError(f"unknown channel {message.channel!r}")
        await chan.send(message)

    def discover_entry_points(self, group: str = "xerxes.channels") -> list[str]:
        """Optional: load channel adapters via Python entry points.

        Returns the list of newly registered channel names.
        """
        try:
            from importlib.metadata import entry_points
        except Exception:
            return []
        added: list[str] = []
        try:
            eps = entry_points(group=group)
        except TypeError:
            eps = entry_points().get(group, [])
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
    """Convenience: start every registry's channels in parallel."""

    async def _run():
        """Await :meth:`ChannelRegistry.start_all` on every supplied registry."""
        await asyncio.gather(*(r.start_all() for r in registries))

    return _run()


__all__ = ["ChannelRegistry", "InboundHandler", "gather_inbound"]
