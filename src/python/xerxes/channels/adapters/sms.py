# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License")
"""SMS adapter — Twilio-shaped webhook + Messages API send."""

from __future__ import annotations

import typing as tp
import urllib.parse

from .._helpers import WebhookChannel, http_post
from ..types import ChannelMessage, MessageDirection


class TwilioSMSChannel(WebhookChannel):
    """Twilio Programmable Messaging adapter.

    Inbound: Twilio POSTs ``application/x-www-form-urlencoded`` with
    fields ``From``, ``To``, ``Body``, ``MessageSid``.
    Outbound: ``POST https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json``
    via HTTP Basic auth.
    """

    name = "sms"

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        *,
        http_client: tp.Any = None,
    ) -> None:
        """Configure the adapter with Twilio API credentials.

        Args:
            account_sid: Twilio Account SID.
            auth_token: Twilio Auth Token (used for HTTP Basic auth).
            from_number: Sender phone number in E.164 format.
            http_client: Optional injected HTTP callable for tests.
        """
        super().__init__()
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Decode ``application/x-www-form-urlencoded`` Twilio callbacks.

        Pulls ``Body``, ``From``, ``To`` and ``MessageSid`` out of the form
        body and emits one :class:`ChannelMessage`.
        """
        try:
            params = dict(urllib.parse.parse_qsl(body.decode("utf-8", errors="ignore")))
        except Exception:
            return []
        if not params:
            return []
        return [
            ChannelMessage(
                text=params.get("Body", ""),
                channel=self.name,
                channel_user_id=params.get("From", ""),
                room_id=params.get("From", ""),
                platform_message_id=params.get("MessageSid", ""),
                direction=MessageDirection.INBOUND,
                metadata={"to": params.get("To", "")},
            )
        ]

    async def _send_outbound(self, message):
        """POST to the Twilio Messages API using Basic auth from SID + token."""
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        body = {"From": self.from_number, "To": message.room_id or message.channel_user_id, "Body": message.text}
        import base64

        cred = base64.b64encode(f"{self.account_sid}:{self.auth_token}".encode()).decode()
        http_post(
            url,
            json_body=body,
            headers={"Authorization": f"Basic {cred}"},
            http_client=self._http,
        )
