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

Speaks WeCom's enterprise messaging API: parses inbound event payloads
and sends text messages via ``cgi-bin/message/send``. Accepts either a
static access token or a ``token_provider`` callable for installs that
refresh the (short-lived) WeCom access token externally.
"""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class WeComChannel(WebhookChannel):
    """WeCom (Enterprise WeChat) adapter."""

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
        """Build the channel.

        Args:
            access_token: Static access token; fallback when
                ``token_provider`` is unavailable.
            agent_id: WeCom agent (application) identifier sent in every
                outbound message payload.
            token_provider: Optional zero-arg callable returning a fresh
                token. Takes precedence over ``access_token``.
            api_base: WeCom API base URL; trailing ``/`` stripped.
            http_client: Optional HTTP client override forwarded to
                ``http_post``.
        """
        super().__init__()
        self.access_token = access_token
        self.agent_id = agent_id
        self.token_provider = token_provider
        self.api_base = api_base.rstrip("/")
        self._http = http_client

    def _resolve_token(self) -> str:
        """Pick the access token for the next outbound call.

        Returns:
            Output of ``token_provider`` when truthy, otherwise the static
            ``access_token``.
        """
        if self.token_provider is not None:
            tok = self.token_provider()
            if tok:
                return tok
        return self.access_token

    def _parse_inbound(self, headers, body):
        """Translate a WeCom event payload into ``ChannelMessage``.

        Tolerates both the official PascalCase (``Content``, ``FromUserName``,
        ``MsgId``) and the snake_case shape used by some bridges. Messages
        with no text body are skipped.

        Args:
            headers: HTTP headers (unused).
            body: Raw JSON webhook body.

        Returns:
            One parsed inbound message, or empty when no text was present.
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
        """Send one text message via ``cgi-bin/message/send``.

        Args:
            message: Outbound message. ``channel_user_id`` (preferred) or
                ``room_id`` becomes ``touser``; ``text`` is the body.
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
