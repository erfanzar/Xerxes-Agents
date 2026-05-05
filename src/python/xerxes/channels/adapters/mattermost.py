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
"""Mattermost channel adapter.

Connects to a Mattermost server for sending and receiving messages.
"""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class MattermostChannel(WebhookChannel):
    """Channel implementation for Mattermost."""

    name = "mattermost"

    def __init__(
        self,
        base_url: str,
        bot_token: str,
        *,
        http_client: tp.Any = None,
    ) -> None:
        """Initialize the Mattermost channel.

        Args:
            base_url (str): IN: Mattermost server base URL.
                OUT: stored with trailing slash removed.
            bot_token (str): IN: bot access token.
                OUT: stored for API authorization.
            http_client (Any): IN: optional HTTP client override.
                OUT: forwarded to ``http_post``.
        """
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.bot_token = bot_token
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Parse a Mattermost webhook payload into ``ChannelMessage``.

        Args:
            headers (dict[str, str]): IN: HTTP headers (unused).
            body (bytes): IN: raw JSON webhook body.

        Returns:
            list[ChannelMessage]: OUT: parsed inbound messages.
        """
        data = parse_json_body(body)
        if not data:
            return []
        return [
            ChannelMessage(
                text=data.get("text", ""),
                channel=self.name,
                channel_user_id=str(data.get("user_id", "")),
                room_id=str(data.get("channel_id", "")),
                platform_message_id=str(data.get("post_id", "")),
                direction=MessageDirection.INBOUND,
                metadata={"team_id": data.get("team_id", "")},
            )
        ]

    async def _send_outbound(self, message):
        """Send a message to a Mattermost channel.

        Args:
            message (ChannelMessage): IN: message to send. ``room_id`` is the
                target channel ID, ``text`` the content, and ``reply_to``
                (if set) the root post ID for threading.
        """
        url = f"{self.base_url}/api/v4/posts"
        body = {"channel_id": message.room_id, "message": message.text}
        if message.reply_to:
            body["root_id"] = message.reply_to
        http_post(
            url,
            json_body=body,
            headers={"Authorization": f"Bearer {self.bot_token}"},
            http_client=self._http,
        )
