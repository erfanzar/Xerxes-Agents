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

Parses Twilio's ``application/x-www-form-urlencoded`` inbound webhook and
sends outbound SMS via the REST ``Messages.json`` endpoint authenticated
with HTTP Basic. Twilio request validation (``X-Twilio-Signature``) is
not enforced here — front the webhook with a verifier when running on
the public Internet.
"""

from __future__ import annotations

import typing as tp
import urllib.parse

from .._helpers import WebhookChannel, http_post
from ..types import ChannelMessage, MessageDirection


class TwilioSMSChannel(WebhookChannel):
    """Twilio SMS adapter."""

    name = "sms"

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        *,
        http_client: tp.Any = None,
    ) -> None:
        """Build the channel.

        Args:
            account_sid: Twilio account SID. Combined with ``auth_token``
                for HTTP Basic auth on outbound calls.
            auth_token: Twilio auth token.
            from_number: Twilio-owned sender number (E.164) used as the
                ``From`` field on outbound messages.
            http_client: Optional HTTP client override forwarded to
                ``http_post``.
        """
        super().__init__()
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Decode Twilio's form-encoded webhook into ``ChannelMessage``.

        Twilio sends ``application/x-www-form-urlencoded`` rather than
        JSON, so we use ``urllib.parse.parse_qsl`` instead of the JSON
        helper. Malformed bodies and empty payloads return an empty list
        so the dispatcher keeps quiet rather than logging an error.

        Args:
            headers: HTTP headers (unused).
            body: Form-encoded request body.

        Returns:
            One parsed inbound message, or empty for unparseable input.
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
        """Send one SMS via ``POST .../Messages.json`` with HTTP Basic auth.

        Args:
            message: Outbound message. ``room_id`` or ``channel_user_id``
                becomes ``To`` (E.164 phone number); ``text`` is ``Body``.
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
