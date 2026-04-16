# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""BlueBubbles iMessage bridge adapter."""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class BlueBubblesChannel(WebhookChannel):
    """BlueBubbles server adapter (iMessage / SMS via macOS bridge).

    Inbound: BlueBubbles server POSTs ``new-message`` events.
    Outbound: ``POST {server}/api/v1/message/text?password=...``.
    """

    name = "bluebubbles"

    def __init__(
        self,
        server_url: str,
        password: str,
        *,
        http_client: tp.Any = None,
    ) -> None:
        """Bind the adapter to a BlueBubbles server.

        Args:
            server_url: Base URL of the BlueBubbles API (trailing ``/`` ok).
            password: Shared password used as a query-string credential.
            http_client: Optional injected HTTP callable for tests.
        """
        super().__init__()
        self.server_url = server_url.rstrip("/")
        self.password = password
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Convert a BlueBubbles ``new-message`` envelope into messages.

        Reads ``text`` / ``body`` as the content, ``chats[0].guid`` as the
        conversation id, and ``handle.address`` as the sender.
        """
        data = parse_json_body(body)
        if not data:
            return []
        msg_data = data.get("data", data)
        text = msg_data.get("text") or msg_data.get("body") or ""
        chat = msg_data.get("chats", [{}])[0] if isinstance(msg_data.get("chats"), list) else msg_data.get("chat", {})
        handle = msg_data.get("handle") or {}
        if not text:
            return []
        return [
            ChannelMessage(
                text=text,
                channel=self.name,
                channel_user_id=handle.get("address", ""),
                room_id=chat.get("guid", ""),
                platform_message_id=msg_data.get("guid", ""),
                direction=MessageDirection.INBOUND,
            )
        ]

    async def _send_outbound(self, message):
        """POST ``message`` to ``/api/v1/message/text`` using the private-api method."""
        url = f"{self.server_url}/api/v1/message/text?password={self.password}"
        body = {"chatGuid": message.room_id, "message": message.text, "method": "private-api"}
        http_post(url, json_body=body, http_client=self._http)
