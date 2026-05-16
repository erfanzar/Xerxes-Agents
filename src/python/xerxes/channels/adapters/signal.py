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

Targets a self-hosted Signal REST bridge (e.g. ``signal-cli-rest-api``).
Inbound payloads follow the bridge's ``envelope``/``dataMessage`` shape;
outbound is a POST to ``/v2/send`` with the registered sender number.
"""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class SignalChannel(WebhookChannel):
    """Self-hosted Signal REST-bridge adapter."""

    name = "signal"

    def __init__(
        self,
        rest_base: str,
        sender_number: str,
        *,
        http_client: tp.Any = None,
    ) -> None:
        """Build the channel.

        Args:
            rest_base: Base URL of the Signal REST bridge; trailing ``/``
                is stripped.
            sender_number: Registered phone number used as ``number`` on
                outbound sends.
            http_client: Optional HTTP client override forwarded to
                ``http_post``.
        """
        super().__init__()
        self.rest_base = rest_base.rstrip("/")
        self.sender_number = sender_number
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Translate a Signal REST-bridge payload into ``ChannelMessage``.

        Handles both the wrapped ``envelope`` form and the bare envelope
        shape, plus both ``dataMessage`` (Signal native) and ``message``
        (some bridges normalise to this). Drops events with no message
        text.

        Args:
            headers: HTTP headers (unused).
            body: Raw JSON webhook body.

        Returns:
            Parsed messages, or empty when no message text is present.
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
        """Send one message via ``POST /v2/send``.

        Args:
            message: Outbound message. ``room_id`` (preferred) or
                ``channel_user_id`` becomes the sole recipient; ``text``
                is the body.
        """
        body = {
            "number": self.sender_number,
            "recipients": [message.room_id or message.channel_user_id],
            "message": message.text,
        }
        http_post(f"{self.rest_base}/v2/send", json_body=body, http_client=self._http)
