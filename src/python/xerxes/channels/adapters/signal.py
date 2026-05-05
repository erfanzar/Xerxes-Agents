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
"""Signal channel adapter.

Connects to a Signal REST API (e.g. signal-cli-rest-api) for messaging.
"""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class SignalChannel(WebhookChannel):
    """Channel implementation for Signal."""

    name = "signal"

    def __init__(
        self,
        rest_base: str,
        sender_number: str,
        *,
        http_client: tp.Any = None,
    ) -> None:
        """Initialize the Signal channel.

        Args:
            rest_base (str): IN: base URL of the Signal REST API.
                OUT: stored with trailing slash removed.
            sender_number (str): IN: phone number of the sending account.
                OUT: used as the ``from`` field on outbound messages.
            http_client (Any): IN: optional HTTP client override.
                OUT: forwarded to ``http_post``.
        """
        super().__init__()
        self.rest_base = rest_base.rstrip("/")
        self.sender_number = sender_number
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Parse a Signal webhook payload into ``ChannelMessage``.

        Args:
            headers (dict[str, str]): IN: HTTP headers (unused).
            body (bytes): IN: raw JSON webhook body.

        Returns:
            list[ChannelMessage]: OUT: parsed inbound messages.
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
        """Send a text message via the Signal REST API.

        Args:
            message (ChannelMessage): IN: message to send. ``room_id`` or
                ``channel_user_id`` is used as the recipient list, and
                ``text`` as the message body.
        """
        body = {
            "number": self.sender_number,
            "recipients": [message.room_id or message.channel_user_id],
            "message": message.text,
        }
        http_post(f"{self.rest_base}/v2/send", json_body=body, http_client=self._http)
