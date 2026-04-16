# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""WeCom (Enterprise WeChat / 企业微信) adapter."""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class WeComChannel(WebhookChannel):
    """WeCom self-built application bot.

    Inbound: WeCom POSTs the (decrypted) callback envelope.
    Outbound: ``POST {api_base}/cgi-bin/message/send?access_token=...``.
    """

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
        """Configure the adapter for a WeCom self-built application.

        Args:
            access_token: Static access token (used when no provider set).
            agent_id: Numeric agent/application id inside the enterprise.
            token_provider: Callable returning the current access token.
            api_base: WeCom open-API base URL.
            http_client: Optional injected HTTP callable for tests.
        """
        super().__init__()
        self.access_token = access_token
        self.agent_id = agent_id
        self.token_provider = token_provider
        self.api_base = api_base.rstrip("/")
        self._http = http_client

    def _resolve_token(self) -> str:
        """Return the current access token, preferring ``token_provider``."""
        if self.token_provider is not None:
            tok = self.token_provider()
            if tok:
                return tok
        return self.access_token

    def _parse_inbound(self, headers, body):
        """Pick content and sender from either XML-derived or JSON fields.

        WeCom's encrypted callback is decoded upstream to JSON; this reads
        either ``Content``/``FromUserName`` (XML shape) or the lowercase
        ``content``/``from_user`` aliases.
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
        """Call ``cgi-bin/message/send`` with an access token on the query string."""
        token = self._resolve_token()
        url = f"{self.api_base}/cgi-bin/message/send?access_token={token}"
        body = {
            "touser": message.channel_user_id or message.room_id,
            "msgtype": "text",
            "agentid": self.agent_id,
            "text": {"content": message.text},
        }
        http_post(url, json_body=body, http_client=self._http)
