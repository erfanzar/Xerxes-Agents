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
"""Home Assistant channel adapter.

Sends outbound messages as persistent notifications and parses inbound
conversation webhook payloads.
"""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class HomeAssistantChannel(WebhookChannel):
    """Channel implementation for Home Assistant."""

    name = "home_assistant"

    def __init__(
        self,
        ha_url: str,
        access_token: str,
        *,
        http_client: tp.Any = None,
    ) -> None:
        """Initialize the Home Assistant channel.

        Args:
            ha_url (str): IN: Home Assistant instance URL.
                OUT: stored with trailing slash removed.
            access_token (str): IN: long-lived access token.
                OUT: stored for API authorization.
            http_client (Any): IN: optional HTTP client override.
                OUT: forwarded to ``http_post``.
        """
        super().__init__()
        self.ha_url = ha_url.rstrip("/")
        self.access_token = access_token
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Parse a Home Assistant webhook payload into ``ChannelMessage``.

        Args:
            headers (dict[str, str]): IN: HTTP headers (unused).
            body (bytes): IN: raw JSON webhook body.

        Returns:
            list[ChannelMessage]: OUT: parsed inbound messages.
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
        """Create a persistent notification in Home Assistant.

        Args:
            message (ChannelMessage): IN: message to send. ``text`` becomes
                the notification body and ``message_id`` the notification ID.
        """
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
