# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""Feishu / Lark adapter (ByteDance enterprise IM)."""

from __future__ import annotations

import json
import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class FeishuChannel(WebhookChannel):
    """Feishu (Lark) bot adapter via app webhook + open-API send.

    Expects a tenant access token to be supplied directly or via an
    ``token_provider`` callable that returns the current token (the
    Feishu token expires every 2 h; the provider lets the caller cache
    + refresh out-of-band).
    """

    name = "feishu"

    def __init__(
        self,
        tenant_access_token: str = "",
        *,
        token_provider: tp.Callable[[], str] | None = None,
        api_base: str = "https://open.feishu.cn",
        http_client: tp.Any = None,
    ) -> None:
        """Configure the adapter with a static or refreshable tenant token.

        Args:
            tenant_access_token: Static tenant access token (used when no
                ``token_provider`` is supplied).
            token_provider: Callable returning the current tenant token.
            api_base: Feishu / Lark open-API base URL.
            http_client: Optional injected HTTP callable for tests.
        """
        super().__init__()
        self.tenant_access_token = tenant_access_token
        self.token_provider = token_provider
        self.api_base = api_base.rstrip("/")
        self._http = http_client

    def _resolve_token(self) -> str:
        """Return the current tenant token, preferring ``token_provider``."""
        if self.token_provider is not None:
            tok = self.token_provider()
            if tok:
                return tok
        return self.tenant_access_token

    def _parse_inbound(self, headers, body):
        """Decode a Feishu event envelope into a :class:`ChannelMessage`.

        Drops ``url_verification`` handshakes. Feishu nests the text body
        as a JSON-encoded string under ``event.message.content``; it is
        parsed here to recover the plain ``text`` field.
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
        """POST a text message to the IM open API addressed by ``chat_id``."""
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
