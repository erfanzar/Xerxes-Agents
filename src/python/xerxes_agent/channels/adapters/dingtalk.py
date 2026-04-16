# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""DingTalk (Alibaba) bot adapter."""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class DingTalkChannel(WebhookChannel):
    """DingTalk custom-bot adapter (incoming webhook + access_token send)."""

    name = "dingtalk"

    def __init__(
        self,
        webhook_url: str,
        *,
        http_client: tp.Any = None,
    ) -> None:
        """Bind the adapter to a DingTalk custom-bot webhook.

        Args:
            webhook_url: Full webhook URL including the ``access_token``
                query parameter provided by DingTalk.
            http_client: Optional injected HTTP callable for tests.
        """
        super().__init__()
        self.webhook_url = webhook_url
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Extract ``text.content`` / ``content``, sender id and conversation id."""
        data = parse_json_body(body)
        if not data:
            return []
        text = (data.get("text") or {}).get("content", "") or data.get("content", "")
        return [
            ChannelMessage(
                text=text,
                channel=self.name,
                channel_user_id=data.get("senderId", "") or data.get("senderStaffId", ""),
                room_id=data.get("conversationId", ""),
                platform_message_id=str(data.get("msgId", "")),
                direction=MessageDirection.INBOUND,
                metadata={"sender_nick": data.get("senderNick", "")},
            )
        ]

    async def _send_outbound(self, message):
        """POST a ``msgtype=text`` envelope back to the bot webhook URL."""
        body = {"msgtype": "text", "text": {"content": message.text}}
        http_post(self.webhook_url, json_body=body, http_client=self._http)
