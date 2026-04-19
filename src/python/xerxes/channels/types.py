# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.

"""Channel message types and direction enum."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class MessageDirection(Enum):
    """Direction of a channel message relative to Xerxes.

    Attributes:
        INBOUND: Message received from the channel (user → agent).
        OUTBOUND: Message sent by the agent (agent → user).
    """

    INBOUND = "inbound"
    OUTBOUND = "outbound"


@dataclass
class ChannelMessage:
    """Normalised cross-channel message envelope.

    A single message structure that all channel adapters convert their
    platform-specific payloads into. The envelope is what the agent
    actually sees, regardless of whether the message originated from
    Telegram, Slack, Email, SMS, etc.

    Attributes:
        text: The plain-text body of the message.
        channel: The name of the channel adapter (e.g. ``"telegram"``).
        user_id: Stable global Xerxes user identifier (after identity
            resolution). May be ``None`` for unauthenticated channels.
        channel_user_id: The platform-specific user identifier
            (e.g. Telegram user ID, Slack user ID).
        room_id: The platform-specific room/chat/conversation identifier.
        reply_to: Optional message ID this message replies to.
        message_id: Stable identifier for this message envelope.
        platform_message_id: The platform-native message ID (for replies,
            edits, deletions).
        attachments: List of attachment descriptors. Each entry is a
            mapping with at least ``type`` and ``url`` or ``data`` keys.
        timestamp: When the message was created/received.
        direction: Whether this message is inbound or outbound.
        metadata: Arbitrary per-platform metadata (thread IDs, mentions,
            etc.) that does not fit the normalised fields.
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
