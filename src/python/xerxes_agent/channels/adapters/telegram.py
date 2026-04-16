# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""Telegram Bot API adapter (webhook + sendMessage)."""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class TelegramChannel(WebhookChannel):
    """Telegram Bot adapter using the official ``Bot API`` HTTPS endpoint.

    Inbound payload spec: https://core.telegram.org/bots/api#update
    Outbound: ``POST https://api.telegram.org/bot<TOKEN>/sendMessage``.
    """

    name = "telegram"

    def __init__(self, token: str, *, http_client: tp.Any = None) -> None:
        """Bind the adapter to a Telegram bot token.

        Args:
            token: Bot token from @BotFather, embedded in the API URL.
            http_client: Optional injected HTTP callable for tests.
        """
        super().__init__()
        self.token = token
        self._http = http_client
        self._base = f"https://api.telegram.org/bot{token}"

    def _parse_inbound(self, headers, body):
        """Extract ``message`` (or ``edited_message``) text, sender and chat.

        Falls back to ``caption`` when the update has no ``text`` (e.g.
        photo captions). Retains ``username`` + ``chat_type`` in metadata.
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
        """Call ``sendMessage``; sets ``reply_to_message_id`` when threading."""
        body = {
            "chat_id": message.room_id or message.channel_user_id,
            "text": message.text,
        }
        if message.reply_to:
            body["reply_to_message_id"] = message.reply_to
        http_post(f"{self._base}/sendMessage", json_body=body, http_client=self._http)
