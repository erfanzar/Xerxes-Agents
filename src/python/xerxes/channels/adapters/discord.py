# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License")
"""Discord adapter — Interactions webhook + REST channel message API."""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class DiscordChannel(WebhookChannel):
    """Discord adapter via the Interactions endpoint URL + REST send.

    Inbound: ``POST /interactions`` (or message-create webhook).
    Outbound: ``POST https://discord.com/api/v10/channels/<id>/messages``.
    """

    name = "discord"

    def __init__(self, bot_token: str, *, http_client: tp.Any = None) -> None:
        """Bind the adapter to a Discord bot.

        Args:
            bot_token: Bot token used as ``Authorization: Bot <token>``.
            http_client: Optional injected HTTP callable for tests.
        """
        super().__init__()
        self.bot_token = bot_token
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Translate a Discord message-create / interaction payload.

        Reads ``content``, ``author.id``, ``channel_id`` and ``id`` from
        either a bare message object or an interaction envelope.
        """
        data = parse_json_body(body)
        if not data:
            return []
        msg = data.get("message") or data
        author = msg.get("author") or {}
        return [
            ChannelMessage(
                text=msg.get("content", ""),
                channel=self.name,
                channel_user_id=str(author.get("id", "")),
                room_id=str(msg.get("channel_id", data.get("channel_id", ""))),
                platform_message_id=str(msg.get("id", "")),
                direction=MessageDirection.INBOUND,
                metadata={"guild_id": msg.get("guild_id", "")},
            )
        ]

    async def _send_outbound(self, message):
        """POST ``message.text`` to the channel, threading via ``message_reference`` when set."""
        url = f"https://discord.com/api/v10/channels/{message.room_id}/messages"
        body = {"content": message.text}
        if message.reply_to:
            body["message_reference"] = {"message_id": message.reply_to}
        http_post(
            url,
            json_body=body,
            headers={"Authorization": f"Bot {self.bot_token}"},
            http_client=self._http,
        )
