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
"""WeCom (企业微信) channel adapter.

Connects to the WeCom API for enterprise messaging.
"""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class WeComChannel(WebhookChannel):
    """Channel implementation for WeCom (Enterprise WeChat)."""

    name = "wecom"

    def __init__(
        self,
        access_token: str = "",
        agent_id: str | int = "",
        *,
        token_provider: tp.Callable[[], str] | None = None,
        api_base: str = "https://qyapi.weixin.qq.com",
        http_client: tp.Any = None,
    ) -> None:
        """Initialize the WeCom channel.

        Args:
            access_token (str): IN: static access token. Defaults to empty.
                OUT: fallback when ``token_provider`` is not given.
            agent_id (str | int): IN: WeCom agent identifier.
                OUT: included in outbound message payloads.
            token_provider (Callable | None): IN: optional callable that
                returns a fresh access token. OUT: takes precedence over the
                static token.
            api_base (str): IN: base URL for WeCom APIs.
                OUT: stored with trailing slash removed.
            http_client (Any): IN: optional HTTP client override.
                OUT: forwarded to ``http_post``.
        """
        super().__init__()
        self.access_token = access_token
        self.agent_id = agent_id
        self.token_provider = token_provider
        self.api_base = api_base.rstrip("/")
        self._http = http_client

    def _resolve_token(self) -> str:
        """Resolve the current access token.

        Returns:
            str: OUT: token from ``token_provider`` if available, otherwise
            the static ``access_token``.
        """
        if self.token_provider is not None:
            tok = self.token_provider()
            if tok:
                return tok
        return self.access_token

    def _parse_inbound(self, headers, body):
        """Parse a WeCom webhook payload into ``ChannelMessage``.

        Args:
            headers (dict[str, str]): IN: HTTP headers (unused).
            body (bytes): IN: raw JSON webhook body.

        Returns:
            list[ChannelMessage]: OUT: parsed inbound messages.
        """
        data = parse_json_body(body)
        if not data:
            return []
        text = data.get("Content") or data.get("content") or ""
        from_user = data.get("FromUserName") or data.get("from_user", "")
        if not text:
            return []
        return [
            ChannelMessage(
                text=text,
                channel=self.name,
                channel_user_id=from_user,
                room_id=from_user,
                platform_message_id=str(data.get("MsgId", "")),
                direction=MessageDirection.INBOUND,
            )
        ]

    async def _send_outbound(self, message):
        """Send a text message via the WeCom API.

        Args:
            message (ChannelMessage): IN: message to send. ``channel_user_id``
                or ``room_id`` is the recipient, and ``text`` the content.
        """
        token = self._resolve_token()
        url = f"{self.api_base}/cgi-bin/message/send?access_token={token}"
        body = {
            "touser": message.channel_user_id or message.room_id,
            "msgtype": "text",
            "agentid": self.agent_id,
            "text": {"content": message.text},
        }
        http_post(url, json_body=body, http_client=self._http)
