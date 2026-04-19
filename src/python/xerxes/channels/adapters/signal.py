# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License")
"""Signal adapter — bridges to a local ``signal-cli`` REST daemon."""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class SignalChannel(WebhookChannel):
    """Signal adapter via ``bbernhard/signal-cli-rest-api`` (or compatible).

    Inbound: REST daemon long-polls and POSTs new envelopes to
    ``/webhooks/signal``.
    Outbound: ``POST {base}/v2/send``.
    """

    name = "signal"

    def __init__(
        self,
        rest_base: str,
        sender_number: str,
        *,
        http_client: tp.Any = None,
    ) -> None:
        """Bind the adapter to a local ``signal-cli-rest-api`` daemon.

        Args:
            rest_base: Base URL of the REST daemon.
            sender_number: Registered Signal phone number (E.164) used
                as the ``From``.
            http_client: Optional injected HTTP callable for tests.
        """
        super().__init__()
        self.rest_base = rest_base.rstrip("/")
        self.sender_number = sender_number
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Unwrap a Signal envelope, extracting ``dataMessage.message`` text.

        Falls back to ``message.message`` (older daemons) and uses the
        source phone number as both sender id and ``room_id``.
        """
        data = parse_json_body(body)
        if not data:
            return []
        envelope = data.get("envelope", {}) if "envelope" in data else data
        msg = envelope.get("dataMessage", {}) or envelope.get("message", {})
        text = msg.get("message") if isinstance(msg, dict) else str(msg)
        if not text:
            return []
        sender = envelope.get("sourceNumber") or envelope.get("source", "")
        return [
            ChannelMessage(
                text=text,
                channel=self.name,
                channel_user_id=sender,
                room_id=sender,
                platform_message_id=str(envelope.get("timestamp", "")),
                direction=MessageDirection.INBOUND,
            )
        ]

    async def _send_outbound(self, message):
        """POST a ``/v2/send`` payload with this sender and a single recipient."""
        body = {
            "number": self.sender_number,
            "recipients": [message.room_id or message.channel_user_id],
            "message": message.text,
        }
        http_post(f"{self.rest_base}/v2/send", json_body=body, http_client=self._http)
