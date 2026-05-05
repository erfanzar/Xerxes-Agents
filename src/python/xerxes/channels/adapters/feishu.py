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
"""Feishu (Lark) channel adapter.

Connects to the Feishu Open Platform for messaging.
"""

from __future__ import annotations

import json
import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class FeishuChannel(WebhookChannel):
    """Channel implementation for Feishu (Lark)."""

    name = "feishu"

    def __init__(
        self,
        tenant_access_token: str = "",
        *,
        token_provider: tp.Callable[[], str] | None = None,
        api_base: str = "https://open.feishu.cn",
        http_client: tp.Any = None,
    ) -> None:
        """Initialize the Feishu channel.

        Args:
            tenant_access_token (str): IN: static tenant access token.
                OUT: fallback token when ``token_provider`` is not given.
            token_provider (Callable | None): IN: optional callable that
                returns a fresh access token. OUT: takes precedence over the
                static token.
            api_base (str): IN: base URL for Feishu open APIs.
                OUT: stored with trailing slash removed.
            http_client (Any): IN: optional HTTP client override.
                OUT: forwarded to ``http_post``.
        """
        super().__init__()
        self.tenant_access_token = tenant_access_token
        self.token_provider = token_provider
        self.api_base = api_base.rstrip("/")
        self._http = http_client

    def _resolve_token(self) -> str:
        """Resolve the current access token.

        Returns:
            str: OUT: token from ``token_provider`` if available, otherwise
            the static ``tenant_access_token``.
        """
        if self.token_provider is not None:
            tok = self.token_provider()
            if tok:
                return tok
        return self.tenant_access_token

    def _parse_inbound(self, headers, body):
        """Parse a Feishu webhook payload into ``ChannelMessage``.

        Args:
            headers (dict[str, str]): IN: HTTP headers (unused).
            body (bytes): IN: raw JSON webhook body.

        Returns:
            list[ChannelMessage]: OUT: parsed inbound messages. Empty for
            URL-verification events.
        """
        data = parse_json_body(body)
        if not data:
            return []
        if data.get("type") == "url_verification":
            return []
        ev = data.get("event") or {}
        message = ev.get("message") or {}
        sender = ev.get("sender") or {}
        text = ""
        try:
            content = json.loads(message.get("content", "{}"))
            text = content.get("text", "")
        except Exception:
            text = message.get("content", "")
        if not text:
            return []
        return [
            ChannelMessage(
                text=text,
                channel=self.name,
                channel_user_id=sender.get("sender_id", {}).get("open_id", ""),
                room_id=message.get("chat_id", ""),
                platform_message_id=message.get("message_id", ""),
                direction=MessageDirection.INBOUND,
            )
        ]

    async def _send_outbound(self, message):
        """Send a text message via the Feishu API.

        Args:
            message (ChannelMessage): IN: message to send. ``room_id`` is used
                as the target chat ID and ``text`` as the message body.
        """
        url = f"{self.api_base}/open-apis/im/v1/messages?receive_id_type=chat_id"
        body = {
            "receive_id": message.room_id,
            "msg_type": "text",
            "content": json.dumps({"text": message.text}),
        }
        http_post(
            url,
            json_body=body,
            headers={"Authorization": f"Bearer {self._resolve_token()}"},
            http_client=self._http,
        )
