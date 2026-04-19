# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License")
"""Matrix adapter — Application Service hookshot style."""

from __future__ import annotations

import time
import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class MatrixChannel(WebhookChannel):
    """Matrix homeserver adapter using a long-lived access token.

    Inbound: appservice or webhook bridge POSTs Matrix events.
    Outbound: ``PUT /_matrix/client/v3/rooms/{roomId}/send/m.room.message/{txnId}``.
    """

    name = "matrix"

    def __init__(
        self,
        homeserver_url: str,
        access_token: str,
        *,
        http_client: tp.Any = None,
    ) -> None:
        """Bind the adapter to a Matrix homeserver.

        Args:
            homeserver_url: Base URL of the Matrix homeserver.
            access_token: Long-lived client access token.
            http_client: Optional injected HTTP callable for tests.
        """
        super().__init__()
        self.homeserver_url = homeserver_url.rstrip("/")
        self.access_token = access_token
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Iterate ``events`` (or a single event), keeping only ``m.room.message``."""
        data = parse_json_body(body)
        events = data.get("events") or [data]
        out = []
        for ev in events:
            if ev.get("type") != "m.room.message":
                continue
            content = ev.get("content") or {}
            out.append(
                ChannelMessage(
                    text=content.get("body", ""),
                    channel=self.name,
                    channel_user_id=ev.get("sender", ""),
                    room_id=ev.get("room_id", ""),
                    platform_message_id=ev.get("event_id", ""),
                    direction=MessageDirection.INBOUND,
                )
            )
        return out

    async def _send_outbound(self, message):
        """PUT an ``m.text`` event into ``message.room_id`` with a fresh txn id."""
        txn = f"xerxes-{int(time.time() * 1000)}"
        url = f"{self.homeserver_url}/_matrix/client/v3/rooms/{message.room_id}/send/m.room.message/{txn}"
        body = {"msgtype": "m.text", "body": message.text}
        http_post(
            url,
            json_body=body,
            headers={"Authorization": f"Bearer {self.access_token}"},
            http_client=self._http,
        )
