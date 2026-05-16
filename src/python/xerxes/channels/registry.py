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

``ChannelRegistry`` is the daemon-facing collection that owns a process's
channels: it gives each channel a name, holds the inbound handler, and
runs ``start_all``/``stop_all`` with best-effort error handling so a
single misconfigured adapter cannot prevent the others from coming up.
``gather_inbound`` is a thin coroutine helper for starting several
registries concurrently.
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
    """Named collection of channels with start/stop lifecycle bookkeeping.

    Tracks which channels have been started so ``start_all`` is idempotent
    and ``stop_all`` only stops what is actually running. Failures during
    start are logged but do not abort the rest of the batch — operators can
    misconfigure one adapter without losing all messaging.
    """

    def __init__(self) -> None:
        """Build an empty registry with no inbound handler set."""
        self._channels: dict[str, Channel] = {}
        self._handler: InboundHandler | None = None
        self._started: set[str] = set()

    def register(self, name: str, channel: Channel) -> None:
        """Add or replace a channel under ``name``.

        Args:
            name: Registry-level identifier. Conventionally matches
                ``channel.name``, but the registry does not enforce it.
            channel: Channel instance to register.
        """
        self._channels[name] = channel

    def unregister(self, name: str) -> None:
        """Remove a channel and forget its started state.

        Does not stop the channel — callers must do that themselves before
        unregistering if they want a clean shutdown.

        Args:
            name: Identifier passed to ``register``.
        """
        self._channels.pop(name, None)
        self._started.discard(name)

    def get(self, name: str) -> Channel | None:
        """Look up a channel by name.

        Args:
            name: Identifier passed to ``register``.

        Returns:
            The channel if registered, otherwise ``None``.
        """
        return self._channels.get(name)

    def all(self) -> dict[str, Channel]:
        """Return a shallow copy of the ``name → channel`` mapping.

        Mutating the returned dict does not affect the registry.
        """
        return dict(self._channels)

    def names(self) -> list[str]:
        """Return the registry's channel names in insertion order."""
        return list(self._channels.keys())

    def set_handler(self, handler: InboundHandler) -> None:
        """Install the single inbound callback used by every channel.

        Must be called before ``start_all`` — channels share one handler so
        the registry can route all inbound traffic into a single agent
        runtime regardless of source platform.

        Args:
            handler: Async callable invoked for every inbound message.
        """
        self._handler = handler

    async def start_all(self) -> None:
        """Start every registered channel that is not already running.

        Each ``Channel.start`` is awaited inside a try/except so one bad
        adapter does not prevent the others from coming up; failures are
        logged at WARNING with the traceback.

        Raises:
            RuntimeError: ``set_handler`` was not called first.
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
        """Stop every currently running channel; swallow per-channel errors.

        Channels whose ``stop`` raises are logged at DEBUG and still removed
        from the started set so the registry remains consistent.
        """
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
        """Route an outbound message to the channel named by ``message.channel``.

        Args:
            message: Outbound message; ``message.channel`` selects the target.

        Raises:
            KeyError: No channel is registered under ``message.channel``.
        """
        chan = self._channels.get(message.channel)
        if chan is None:
            raise KeyError(f"unknown channel {message.channel!r}")
        await chan.send(message)

    def discover_entry_points(self, group: str = "xerxes.channels") -> list[str]:
        """Auto-register channels declared via ``importlib.metadata`` entry points.

        Each entry point must be a zero-arg factory returning a ``Channel``
        instance. Factories that fail to import or return a non-``Channel``
        are skipped with a logged warning so a single broken plugin cannot
        crash startup.

        Args:
            group: Entry-point group name. Defaults to ``"xerxes.channels"``.

        Returns:
            Names of channels that were loaded and registered successfully.
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
    """Start ``start_all`` on every registry concurrently.

    Args:
        *registries: One or more registries whose channels should come up
            in parallel.

    Returns:
        A coroutine that resolves when every registry finishes its
        ``start_all`` (including channels that failed to start individually).
    """

    async def _run():
        """Run ``start_all`` on every registry concurrently via ``asyncio.gather``."""
        await asyncio.gather(*(r.start_all() for r in registries))

    return _run()


__all__ = ["ChannelRegistry", "InboundHandler", "gather_inbound"]
