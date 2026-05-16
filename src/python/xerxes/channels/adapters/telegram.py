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
"""Telegram channel adapter.

Wraps the Telegram Bot API: parses inbound webhook payloads into
``ChannelMessage`` objects and sends replies via ``sendMessage``. The
adapter intentionally stays unaware of authentication — webhook
authenticity and sender allowlisting happen in
``channels.telegram_gateway``.
"""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class TelegramChannel(WebhookChannel):
    """Telegram Bot API adapter."""

    name = "telegram"

    def __init__(self, token: str, *, http_client: tp.Any = None, accept_edited_messages: bool = False) -> None:
        """Build the channel.

        Args:
            token: Telegram bot token; embedded in the Bot API base URL.
            http_client: Optional HTTP client override forwarded to
                ``http_post``.
            accept_edited_messages: When ``True``, ``edited_message`` updates
                are parsed and dispatched. Off by default to prevent
                replay-by-edit: a malicious sender can otherwise edit an
                old message repeatedly to keep re-triggering agent runs.
        """
        super().__init__()
        self.token = token
        self._http = http_client
        self._accept_edited = accept_edited_messages
        self._base = f"https://api.telegram.org/bot{token}"

    def _parse_inbound(self, headers, body):
        """Translate a Telegram update into ``ChannelMessage`` instances.

        Handles both ``message`` and (when ``accept_edited_messages``)
        ``edited_message`` updates. Captions of media messages are also
        used so a photo with an attached caption still produces text the
        agent can act on. Returns an empty list when there is no message
        envelope at all.

        Args:
            headers: HTTP headers (unused; provided for interface
                compatibility).
            body: Raw JSON webhook body.

        Returns:
            A single-element list with the parsed message, or empty when
            the update carries nothing relevant.
        """
        data = parse_json_body(body)
        message = data.get("message") or {}
        if not message and self._accept_edited:
            message = data.get("edited_message") or {}
        chat = message.get("chat") or {}
        sender = message.get("from") or {}
        if not message:
            return []
        text = message.get("text") or message.get("caption") or ""
        return [
            ChannelMessage(
                text=text,
                channel=self.name,
                channel_user_id=str(sender.get("id", "")),
                room_id=str(chat.get("id", "")),
                platform_message_id=str(message.get("message_id", "")),
                direction=MessageDirection.INBOUND,
                metadata={
                    "username": sender.get("username", ""),
                    "first_name": sender.get("first_name", ""),
                    "last_name": sender.get("last_name", ""),
                    "chat_type": chat.get("type", ""),
                    "chat_title": chat.get("title", ""),
                    "thread_id": str(message.get("message_thread_id", "")),
                },
            )
        ]

    def get_updates(self, *, offset: int | None = None, timeout: int = 30) -> dict[str, tp.Any]:
        """Long-poll the Telegram Bot API for new updates.

        Restricts the ``allowed_updates`` list to ``message`` (and
        ``edited_message`` when configured) so the bot does not receive
        callback queries or inline-mode traffic it cannot service.

        Args:
            offset: Last-seen update id + 1; tells Telegram which updates
                to skip. ``None`` returns the oldest unread batch.
            timeout: Long-poll timeout in seconds; Telegram will hold the
                connection that long before responding empty.

        Returns:
            The raw Telegram Bot API response.
        """
        allowed = ["message", "edited_message"] if self._accept_edited else ["message"]
        body: dict[str, tp.Any] = {"timeout": timeout, "allowed_updates": allowed}
        if offset is not None:
            body["offset"] = offset
        return http_post(f"{self._base}/getUpdates", json_body=body, http_client=self._http, timeout=timeout + 5)

    async def send_text(self, *, chat_id: str, text: str, reply_to: str | None = None) -> dict[str, tp.Any]:
        """Call ``sendMessage`` and return the raw Bot API response.

        Args:
            chat_id: Target chat id.
            text: Message body.
            reply_to: Optional message id to quote-reply to.
        """
        body: dict[str, tp.Any] = {"chat_id": chat_id, "text": text}
        if reply_to:
            body["reply_to_message_id"] = reply_to
        return http_post(f"{self._base}/sendMessage", json_body=body, http_client=self._http)

    async def edit_text(self, *, chat_id: str, message_id: str, text: str) -> dict[str, tp.Any]:
        """Replace the text of a previously sent message via ``editMessageText``.

        Used by ``TelegramAgentGateway`` to stream preview updates without
        spamming the chat with a new message per chunk.
        """
        body = {"chat_id": chat_id, "message_id": message_id, "text": text}
        return http_post(f"{self._base}/editMessageText", json_body=body, http_client=self._http)

    async def _send_outbound(self, message):
        """Deliver one outbound message via ``send_text``.

        Args:
            message: Outbound message. ``room_id`` (preferred) or
                ``channel_user_id`` selects the chat; ``text`` is the body;
                ``reply_to`` becomes ``reply_to_message_id`` when set.
        """
        await self.send_text(
            chat_id=message.room_id or message.channel_user_id,
            text=message.text,
            reply_to=message.reply_to,
        )
