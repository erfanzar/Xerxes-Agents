# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License")
"""Slack Events API adapter."""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class SlackChannel(WebhookChannel):
    """Slack adapter using Events API (``POST /events``) + chat.postMessage.

    Provide a static bot token *or* an ``oauth_client`` plus
    ``install_id`` to fetch a token dynamically.
    """

    name = "slack"

    def __init__(
        self,
        bot_token: str = "",
        *,
        oauth_client: tp.Any = None,
        install_id: str = "default",
        http_client: tp.Any = None,
    ) -> None:
        """Configure Slack credentials (static token or OAuth lookup).

        Args:
            bot_token: Static ``xoxb-`` token; takes precedence when set.
            oauth_client: :class:`OAuthClient` used when ``bot_token`` is
                empty; fetched via ``get_valid_token(install_id)``.
            install_id: Workspace identifier when using OAuth storage.
            http_client: Optional injected HTTP callable for tests.
        """
        super().__init__()
        self.bot_token = bot_token
        self.oauth_client = oauth_client
        self.install_id = install_id
        self._http = http_client

    def _resolve_token(self) -> str:
        """Return the Slack bot token, preferring the static value over OAuth."""
        if self.bot_token:
            return self.bot_token
        if self.oauth_client is not None:
            tok = self.oauth_client.get_valid_token(self.install_id)
            if tok and tok.access_token:
                return tok.access_token
        raise RuntimeError("Slack bot token unavailable")

    def _parse_inbound(self, headers, body):
        """Translate a Slack Events API envelope to a :class:`ChannelMessage`.

        Drops ``url_verification`` challenges and bot-authored events, and
        keeps only ``message``/``app_mention`` events. Retains ``team_id``
        and ``thread_ts`` in metadata.
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
        """Call ``chat.postMessage``; uses ``reply_to`` as ``thread_ts``."""
        body = {"channel": message.room_id, "text": message.text}
        if message.reply_to:
            body["thread_ts"] = message.reply_to
        http_post(
            "https://slack.com/api/chat.postMessage",
            json_body=body,
            headers={"Authorization": f"Bearer {self._resolve_token()}"},
            http_client=self._http,
        )
