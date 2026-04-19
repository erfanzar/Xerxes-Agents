# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License")
"""Home Assistant conversation adapter."""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class HomeAssistantChannel(WebhookChannel):
    """Home Assistant conversation API adapter.

    Inbound: HA fires a webhook (``automation -> rest``) on each user
    voice/text request.
    Outbound: ``POST {ha_url}/api/conversation/process`` (or
    ``persistent_notification.create`` for room messages).
    """

    name = "home_assistant"

    def __init__(
        self,
        ha_url: str,
        access_token: str,
        *,
        http_client: tp.Any = None,
    ) -> None:
        """Bind the adapter to a Home Assistant instance.

        Args:
            ha_url: Base URL of the Home Assistant REST API.
            access_token: Long-lived access token used as ``Bearer``.
            http_client: Optional injected HTTP callable for tests.
        """
        super().__init__()
        self.ha_url = ha_url.rstrip("/")
        self.access_token = access_token
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Pick ``text`` / ``input.text`` / ``message`` out of the HA payload.

        Also retains ``language`` in metadata for downstream NLU routing.
        """
        data = parse_json_body(body)
        if not data:
            return []
        text = data.get("text") or data.get("input", {}).get("text", "") or data.get("message", "")
        if not text:
            return []
        return [
            ChannelMessage(
                text=text,
                channel=self.name,
                channel_user_id=str(data.get("user_id", "")),
                room_id=str(data.get("conversation_id", "")),
                platform_message_id=str(data.get("event_id", "")),
                direction=MessageDirection.INBOUND,
                metadata={"language": data.get("language", "en")},
            )
        ]

    async def _send_outbound(self, message):
        """Post ``message`` via the ``persistent_notification.create`` service."""
        url = f"{self.ha_url}/api/services/persistent_notification/create"
        body = {
            "title": "Xerxes",
            "message": message.text,
            "notification_id": message.message_id,
        }
        http_post(
            url,
            json_body=body,
            headers={"Authorization": f"Bearer {self.access_token}"},
            http_client=self._http,
        )
