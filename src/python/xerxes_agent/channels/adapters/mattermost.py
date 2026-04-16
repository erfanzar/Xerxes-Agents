# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""Mattermost adapter — outgoing webhook + REST post."""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class MattermostChannel(WebhookChannel):
    """Mattermost outgoing-webhook ingest + REST post-message outbound."""

    name = "mattermost"

    def __init__(
        self,
        base_url: str,
        bot_token: str,
        *,
        http_client: tp.Any = None,
    ) -> None:
        """Bind the adapter to a Mattermost server.

        Args:
            base_url: Base URL of the Mattermost instance.
            bot_token: Bot personal access token used as ``Bearer``.
            http_client: Optional injected HTTP callable for tests.
        """
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.bot_token = bot_token
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Map an outgoing-webhook form payload to a :class:`ChannelMessage`.

        Preserves ``team_id`` in metadata so replies can be routed back to
        the right team.
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
        """POST to ``/api/v4/posts``; uses ``reply_to`` as ``root_id`` for threading."""
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
