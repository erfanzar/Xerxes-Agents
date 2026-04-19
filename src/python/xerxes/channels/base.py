# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.
"""Abstract base class for messaging channels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from .types import ChannelMessage

InboundHandler = Callable[[ChannelMessage], Awaitable[None]]


class Channel(ABC):
    """Base class every channel adapter implements.

    A channel bridges an external messaging platform into Xerxes. The
    runtime starts the channel, registers an inbound handler that
    dispatches messages to the agent, and uses :meth:`send` to deliver
    agent responses back out.

    Channel implementations are responsible for:

    - Translating platform payloads into :class:`ChannelMessage` envelopes
      on the inbound path.
    - Translating outbound :class:`ChannelMessage` envelopes back into
      platform payloads on the outbound path.
    - Lifecycle management (connection, reconnection, graceful shutdown).
    - Surfacing platform errors via standard logging or hooks.

    Attributes:
        name: Stable identifier for this channel
            (e.g. ``"telegram"``, ``"slack"``).
    """

    name: str = ""

    @abstractmethod
    async def start(self, on_inbound: InboundHandler) -> None:
        """Start the channel and begin dispatching inbound messages.

        Implementations should establish the platform connection,
        subscribe to relevant events/webhooks, and call ``on_inbound``
        for every incoming :class:`ChannelMessage`.

        Args:
            on_inbound: Coroutine called for each inbound envelope.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel and release its resources.

        Implementations should close network connections, cancel any
        background tasks, and flush pending outbound messages.
        """

    @abstractmethod
    async def send(self, message: ChannelMessage) -> None:
        """Deliver an outbound message to the platform.

        The envelope's ``room_id`` (and optionally ``channel_user_id``
        and ``reply_to``) determines the destination on the platform.

        Args:
            message: The outbound :class:`ChannelMessage` to send.
        """
