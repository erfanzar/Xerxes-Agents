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

Parses Open Platform event-subscription payloads and sends text messages
via the ``im/v1/messages`` endpoint. Supports either a static
``tenant_access_token`` or a ``token_provider`` callback for installs
that need to refresh the token periodically.
"""

from __future__ import annotations

import json
import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class FeishuChannel(WebhookChannel):
    """Feishu (Lark) Open Platform adapter."""

    name = "feishu"

    def __init__(
        self,
        tenant_access_token: str = "",
        *,
        token_provider: tp.Callable[[], str] | None = None,
        api_base: str = "https://open.feishu.cn",
        http_client: tp.Any = None,
    ) -> None:
        """Build the channel.

        Args:
            tenant_access_token: Static tenant access token; used when
                ``token_provider`` is not supplied or returns falsy.
            token_provider: Optional zero-arg callable returning a fresh
                token. Called on every outbound send so the operator can
                centralise refresh.
            api_base: Open Platform base URL; trailing ``/`` stripped.
            http_client: Optional HTTP client override forwarded to
                ``http_post``.
        """
        super().__init__()
        self.tenant_access_token = tenant_access_token
        self.token_provider = token_provider
        self.api_base = api_base.rstrip("/")
        self._http = http_client

    def _resolve_token(self) -> str:
        """Pick the access token for the next outbound call.

        Returns:
            Output of ``token_provider`` when it returns a truthy value,
            otherwise the static ``tenant_access_token``.
        """
        if self.token_provider is not None:
            tok = self.token_provider()
            if tok:
                return tok
        return self.tenant_access_token

    def _parse_inbound(self, headers, body):
        """Translate a Feishu event payload into ``ChannelMessage`` instances.

        Drops URL-verification challenges. Feishu wraps message content as
        a JSON string under ``event.message.content``; we decode it and
        fall back to the raw value if decoding fails. Messages with no
        text are skipped.

        Args:
            headers: HTTP headers (unused).
            body: Raw JSON webhook body.

        Returns:
            Parsed messages, empty when nothing meaningful was present.
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
        """Send one text message via ``im/v1/messages?receive_id_type=chat_id``.

        Feishu requires ``content`` to be a JSON-encoded string rather than
        a nested object, so we encode ``text`` explicitly.

        Args:
            message: Outbound message. ``room_id`` is the chat id and
                ``text`` the body.
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
