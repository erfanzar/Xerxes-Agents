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

Connects to the Telegram Bot API for sending and receiving messages.
"""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class TelegramChannel(WebhookChannel):
    """Channel implementation for Telegram."""

    name = "telegram"

    def __init__(self, token: str, *, http_client: tp.Any = None) -> None:
        """Initialize the Telegram channel.

        Args:
            token (str): IN: Telegram bot token.
                OUT: used to build the Bot API base URL.
            http_client (Any): IN: optional HTTP client override.
                OUT: forwarded to ``http_post``.
        """
        super().__init__()
        self.token = token
        self._http = http_client
        self._base = f"https://api.telegram.org/bot{token}"

    def _parse_inbound(self, headers, body):
        """Parse a Telegram webhook payload into ``ChannelMessage``.

        Args:
            headers (dict[str, str]): IN: HTTP headers (unused).
            body (bytes): IN: raw JSON webhook body.

        Returns:
            list[ChannelMessage]: OUT: parsed inbound messages.
        """
        data = parse_json_body(body)
        message = data.get("message") or data.get("edited_message") or {}
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
                metadata={"username": sender.get("username", ""), "chat_type": chat.get("type", "")},
            )
        ]

    async def _send_outbound(self, message):
        """Send a text message via the Telegram Bot API.

        Args:
            message (ChannelMessage): IN: message to send. ``room_id`` or
                ``channel_user_id`` is the target chat, ``text`` the content,
                and ``reply_to`` (if set) the message ID to reply to.
        """
        body = {
            "chat_id": message.room_id or message.channel_user_id,
            "text": message.text,
        }
        if message.reply_to:
            body["reply_to_message_id"] = message.reply_to
        http_post(f"{self._base}/sendMessage", json_body=body, http_client=self._http)
