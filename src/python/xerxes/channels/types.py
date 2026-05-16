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

Defines ``MessageDirection`` and ``ChannelMessage`` — the platform-neutral
message representation used by every adapter, the registry, the webhook
dispatcher, and the daemon's channel runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class MessageDirection(Enum):
    """Whether a ``ChannelMessage`` came from the user or is bound for them.

    Attributes:
        INBOUND: User → agent. Produced by adapters parsing webhooks.
        OUTBOUND: Agent → user. Produced by the runtime when replying.
    """

    INBOUND = "inbound"
    OUTBOUND = "outbound"


@dataclass
class ChannelMessage:
    """One message moving through a channel, in either direction.

    The same shape is used inbound and outbound; the ``direction`` field
    disambiguates. Adapters fill ``platform_message_id`` from the upstream
    service while ``message_id`` is always a Xerxes-generated UUID — the
    two are never the same and serve different purposes (audit trail vs.
    reply threading).

    Attributes:
        text: Message body text.
        channel: Channel name (matches ``Channel.name`` of the adapter).
        user_id: Resolved global Xerxes user id, populated by
            ``IdentityResolver`` for inbound messages. Optional.
        channel_user_id: Platform-specific user id (Slack ``U..``,
            Telegram numeric id, email address, etc.).
        room_id: Conversation/room/thread identifier on the upstream
            platform. Used as the reply destination for outbound messages.
        reply_to: Upstream message id this message is replying to. Maps to
            ``thread_ts`` on Slack, ``reply_to_message_id`` on Telegram, etc.
        message_id: Xerxes-side UUID, generated automatically. Stable for
            the lifetime of the dataclass.
        platform_message_id: The id assigned by the external platform.
            Empty for outbound messages that have not yet been sent.
        attachments: Per-attachment metadata dicts; structure is
            adapter-defined.
        timestamp: Local wall-clock time at construction.
        direction: ``INBOUND`` or ``OUTBOUND``.
        metadata: Free-form key/value bag for per-platform extras
            (Telegram ``chat_type``, Slack ``team_id``, ...).
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
