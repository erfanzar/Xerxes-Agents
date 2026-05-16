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
"""Base channel abstraction.

Defines the ``Channel`` ABC and the ``InboundHandler`` callable type that
every concrete channel must satisfy. The registry, webhook dispatcher,
and Telegram gateway all interact with channels through this interface,
so adapters never see daemon internals and the daemon never sees
platform SDKs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from .types import ChannelMessage

InboundHandler = Callable[[ChannelMessage], Awaitable[None]]


class Channel(ABC):
    """Abstract base for every messaging channel.

    A channel owns its transport layer: it starts/stops connections, sends
    outbound messages, and pushes inbound traffic into the agent runtime via
    the ``on_inbound`` callback supplied to ``start``. The class attribute
    ``name`` is the stable identifier used by ``ChannelMessage.channel`` and
    by registries; subclasses set it to ``"slack"``, ``"telegram"``, etc.
    """

    name: str = ""

    @abstractmethod
    async def start(self, on_inbound: InboundHandler) -> None:
        """Open the transport and register the inbound handler.

        Args:
            on_inbound: Async callback invoked once per inbound
                ``ChannelMessage``. Implementations must store it and call it
                for every parsed inbound update.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Close the transport and release any underlying resources.

        After ``stop`` the channel should reject further inbound traffic
        (or simply drop it) until ``start`` is called again.
        """

    @abstractmethod
    async def send(self, message: ChannelMessage) -> None:
        """Transmit one outbound message through the platform.

        Args:
            message: The message to deliver. The channel decides which fields
                map to the platform's recipient, threading, and attachment
                semantics.
        """
