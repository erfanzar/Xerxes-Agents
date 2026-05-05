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
"""Matrix channel adapter.

Connects to a Matrix homeserver for sending and receiving room messages.
"""

from __future__ import annotations

import time
import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class MatrixChannel(WebhookChannel):
    """Channel implementation for Matrix."""

    name = "matrix"

    def __init__(
        self,
        homeserver_url: str,
        access_token: str,
        *,
        http_client: tp.Any = None,
    ) -> None:
        """Initialize the Matrix channel.

        Args:
            homeserver_url (str): IN: base URL of the Matrix homeserver.
                OUT: stored with trailing slash removed.
            access_token (str): IN: Matrix access token for the bot user.
                OUT: stored for authorizing API requests.
            http_client (Any): IN: optional HTTP client override.
                OUT: forwarded to ``http_post``.
        """
        super().__init__()
        self.homeserver_url = homeserver_url.rstrip("/")
        self.access_token = access_token
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Parse a Matrix webhook payload into ``ChannelMessage``.

        Args:
            headers (dict[str, str]): IN: HTTP headers (unused).
            body (bytes): IN: raw JSON webhook body.

        Returns:
            list[ChannelMessage]: OUT: parsed inbound messages. Only
            ``m.room.message`` events are processed.
        """
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
        """Send a text message to a Matrix room.

        Args:
            message (ChannelMessage): IN: message to send. ``room_id`` is the
                target room and ``text`` the message body.
        """
        txn = f"xerxes-{int(time.time() * 1000)}"
        url = f"{self.homeserver_url}/_matrix/client/v3/rooms/{message.room_id}/send/m.room.message/{txn}"
        body = {"msgtype": "m.text", "body": message.text}
        http_post(
            url,
            json_body=body,
            headers={"Authorization": f"Bearer {self.access_token}"},
            http_client=self._http,
        )
