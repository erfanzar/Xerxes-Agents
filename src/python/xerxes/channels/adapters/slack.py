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
"""Slack channel adapter.

Connects to Slack via bot token or OAuth for sending and receiving messages.
"""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class SlackChannel(WebhookChannel):
    """Channel implementation for Slack."""

    name = "slack"

    def __init__(
        self,
        bot_token: str = "",
        *,
        oauth_client: tp.Any = None,
        install_id: str = "default",
        http_client: tp.Any = None,
    ) -> None:
        """Initialize the Slack channel.

        Args:
            bot_token (str): IN: static Slack bot token. Defaults to empty.
                OUT: used directly when provided.
            oauth_client (Any): IN: optional ``OAuthClient``-like object with
                ``get_valid_token(install_id)``. OUT: used to resolve a token
                when ``bot_token`` is empty.
            install_id (str): IN: installation identifier for OAuth lookup.
                Defaults to "default".
            http_client (Any): IN: optional HTTP client override.
                OUT: forwarded to ``http_post``.
        """
        super().__init__()
        self.bot_token = bot_token
        self.oauth_client = oauth_client
        self.install_id = install_id
        self._http = http_client

    def _resolve_token(self) -> str:
        """Resolve the Slack bot token to use for API calls.

        Returns:
            str: OUT: static ``bot_token`` if available, otherwise the access
            token from ``oauth_client``.

        Raises:
            RuntimeError: If no token can be resolved.
        """
        if self.bot_token:
            return self.bot_token
        if self.oauth_client is not None:
            tok = self.oauth_client.get_valid_token(self.install_id)
            if tok and tok.access_token:
                return tok.access_token
        raise RuntimeError("Slack bot token unavailable")

    def _parse_inbound(self, headers, body):
        """Parse a Slack Events API payload into ``ChannelMessage``.

        Args:
            headers (dict[str, str]): IN: HTTP headers (unused).
            body (bytes): IN: raw JSON webhook body.

        Returns:
            list[ChannelMessage]: OUT: parsed inbound messages. Empty for
            URL verifications and bot messages.
        """
        data = parse_json_body(body)
        if data.get("type") == "url_verification":
            return []
        ev = data.get("event") or {}
        if ev.get("type") not in ("message", "app_mention"):
            return []
        if ev.get("bot_id"):
            return []
        return [
            ChannelMessage(
                text=ev.get("text", ""),
                channel=self.name,
                channel_user_id=str(ev.get("user", "")),
                room_id=str(ev.get("channel", "")),
                platform_message_id=str(ev.get("ts", "")),
                direction=MessageDirection.INBOUND,
                metadata={"team_id": data.get("team_id", ""), "thread_ts": ev.get("thread_ts", "")},
            )
        ]

    async def _send_outbound(self, message):
        """Send a message to a Slack channel.

        Args:
            message (ChannelMessage): IN: message to send. ``room_id`` is the
                target channel, ``text`` the content, and ``reply_to`` (if set)
                the thread timestamp.
        """
        body = {"channel": message.room_id, "text": message.text}
        if message.reply_to:
            body["thread_ts"] = message.reply_to
        http_post(
            "https://slack.com/api/chat.postMessage",
            json_body=body,
            headers={"Authorization": f"Bearer {self._resolve_token()}"},
            http_client=self._http,
        )
