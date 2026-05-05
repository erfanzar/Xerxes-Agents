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
"""Twilio SMS channel adapter.

Sends and receives SMS messages via the Twilio API.
"""

from __future__ import annotations

import typing as tp
import urllib.parse

from .._helpers import WebhookChannel, http_post
from ..types import ChannelMessage, MessageDirection


class TwilioSMSChannel(WebhookChannel):
    """Channel implementation for Twilio SMS."""

    name = "sms"

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        *,
        http_client: tp.Any = None,
    ) -> None:
        """Initialize the Twilio SMS channel.

        Args:
            account_sid (str): IN: Twilio account SID.
                OUT: stored for API authentication.
            auth_token (str): IN: Twilio auth token.
                OUT: stored for API authentication.
            from_number (str): IN: sender phone number.
                OUT: used as the ``From`` field on outbound messages.
            http_client (Any): IN: optional HTTP client override.
                OUT: forwarded to ``http_post``.
        """
        super().__init__()
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Parse a Twilio webhook payload into ``ChannelMessage``.

        Args:
            headers (dict[str, str]): IN: HTTP headers (unused).
            body (bytes): IN: application/x-www-form-urlencoded body.

        Returns:
            list[ChannelMessage]: OUT: parsed inbound messages.
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
        """Send an SMS via the Twilio API.

        Args:
            message (ChannelMessage): IN: message to send. ``room_id`` or
                ``channel_user_id`` is the recipient, and ``text`` the body.
        """
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
