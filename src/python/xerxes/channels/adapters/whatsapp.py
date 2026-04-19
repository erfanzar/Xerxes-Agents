# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License")
"""WhatsApp Business Cloud API adapter."""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class WhatsAppChannel(WebhookChannel):
    """WhatsApp Cloud API (Meta Graph) adapter.

    Inbound: Graph webhook event payload.
    Outbound: ``POST https://graph.facebook.com/v18.0/{phone_id}/messages``.
    """

    name = "whatsapp"

    def __init__(
        self,
        access_token: str,
        phone_number_id: str,
        *,
        http_client: tp.Any = None,
        api_version: str = "v18.0",
    ) -> None:
        """Configure the adapter for the Meta Graph Cloud API.

        Args:
            access_token: Long-lived Graph API access token.
            phone_number_id: Business phone number id as issued by Meta.
            http_client: Optional injected HTTP callable for tests.
            api_version: Graph API version segment (e.g. ``v18.0``).
        """
        super().__init__()
        self.access_token = access_token
        self.phone_number_id = phone_number_id
        self.api_version = api_version
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Walk ``entry[].changes[].value.messages[]`` and emit one message each.

        Supports both plain ``text.body`` content and button replies
        (``button.text``); sender phone number is used as both user id
        and ``room_id``.
        """
        data = parse_json_body(body)
        out = []
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value") or {}
                for msg in value.get("messages", []):
                    text = (msg.get("text") or {}).get("body", "") or msg.get("button", {}).get("text", "")
                    out.append(
                        ChannelMessage(
                            text=text,
                            channel=self.name,
                            channel_user_id=msg.get("from", ""),
                            room_id=msg.get("from", ""),
                            platform_message_id=msg.get("id", ""),
                            direction=MessageDirection.INBOUND,
                        )
                    )
        return out

    async def _send_outbound(self, message):
        """POST a ``type=text`` WhatsApp message via the Graph API."""
        url = f"https://graph.facebook.com/{self.api_version}/{self.phone_number_id}/messages"
        body = {
            "messaging_product": "whatsapp",
            "to": message.room_id or message.channel_user_id,
            "type": "text",
            "text": {"body": message.text},
        }
        http_post(
            url,
            json_body=body,
            headers={"Authorization": f"Bearer {self.access_token}"},
            http_client=self._http,
        )
