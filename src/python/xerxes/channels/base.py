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
all concrete channel implementations must satisfy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from .types import ChannelMessage

InboundHandler = Callable[[ChannelMessage], Awaitable[None]]


class Channel(ABC):
    """Abstract base class for all messaging channels.

    A channel is responsible for starting/stopping its transport layer and
    for sending outbound messages. Inbound messages are delivered via the
    ``on_inbound`` callback supplied to ``start``.
    """

    name: str = ""

    @abstractmethod
    async def start(self, on_inbound: InboundHandler) -> None:
        """Start the channel and register the inbound handler.

        Args:
            on_inbound (InboundHandler): IN: async callback invoked for each
                inbound ``ChannelMessage``. OUT: stored and called by the
                channel transport.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel and release any transport resources."""

    @abstractmethod
    async def send(self, message: ChannelMessage) -> None:
        """Send an outbound message.

        Args:
            message (ChannelMessage): IN: message to transmit.
        """
