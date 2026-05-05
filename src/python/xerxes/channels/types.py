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
"""Core data types for the channels package.

Defines ``MessageDirection`` and ``ChannelMessage`` — the primary message
representation used across all channel adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class MessageDirection(Enum):
    """Direction of a channel message."""

    INBOUND = "inbound"
    OUTBOUND = "outbound"


@dataclass
class ChannelMessage:
    """A single message exchanged through a channel.

    Attributes:
        text: Message body text.
        channel: Channel name / identifier.
        user_id: Global user ID (optional).
        channel_user_id: Platform-specific user ID (optional).
        room_id: Conversation / room / thread identifier (optional).
        reply_to: ID of the message being replied to (optional).
        message_id: Unique message ID generated automatically.
        platform_message_id: ID assigned by the external platform (optional).
        attachments: List of attachment metadata dicts.
        timestamp: UTC timestamp of message creation.
        direction: Whether the message is inbound or outbound.
        metadata: Free-form key-value metadata.
    """

    text: str
    channel: str
    user_id: str | None = None
    channel_user_id: str | None = None
    room_id: str | None = None
    reply_to: str | None = None
    message_id: str = field(default_factory=lambda: str(uuid4()))
    platform_message_id: str | None = None
    attachments: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    direction: MessageDirection = MessageDirection.INBOUND
    metadata: dict[str, Any] = field(default_factory=dict)
