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
"""Home Assistant channel adapter.

Inbound: parses Home Assistant's conversation webhook (used by voice
assistants and the conversation integration). Outbound: pushes replies
as ``persistent_notification.create`` service calls so the user sees
them in the HA UI. Authentication uses a long-lived access token.
"""

from __future__ import annotations

import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection


class HomeAssistantChannel(WebhookChannel):
    """Home Assistant adapter."""

    name = "home_assistant"

    def __init__(
        self,
        ha_url: str,
        access_token: str,
        *,
        http_client: tp.Any = None,
    ) -> None:
        """Build the channel.

        Args:
            ha_url: Base URL of the Home Assistant instance; trailing ``/``
                stripped.
            access_token: Long-lived access token used for ``Authorization``.
            http_client: Optional HTTP client override forwarded to
                ``http_post``.
        """
        super().__init__()
        self.ha_url = ha_url.rstrip("/")
        self.access_token = access_token
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Translate a Home Assistant conversation webhook into ``ChannelMessage``.

        Accepts three text shapes: top-level ``text``, nested
        ``input.text`` (voice pipelines), and bare ``message``. Empty
        payloads or events with no text are silently dropped. The
        conversation language is preserved on ``metadata['language']``.

        Args:
            headers: HTTP headers (unused).
            body: Raw JSON webhook body.

        Returns:
            One parsed inbound message, or empty when there is no text.
        """
        data = parse_json_body(body)
        if not data:
            return []
        text = data.get("text") or data.get("input", {}).get("text", "") or data.get("message", "")
        if not text:
            return []
        return [
            ChannelMessage(
                text=text,
                channel=self.name,
                channel_user_id=str(data.get("user_id", "")),
                room_id=str(data.get("conversation_id", "")),
                platform_message_id=str(data.get("event_id", "")),
                direction=MessageDirection.INBOUND,
                metadata={"language": data.get("language", "en")},
            )
        ]

    async def _send_outbound(self, message):
        """Create a Home Assistant persistent notification with the reply.

        Persistent notifications surface in the HA UI rather than pushing
        anywhere — this is the broadest delivery channel HA exposes
        without configuring a specific notify integration.

        Args:
            message: Outbound message. ``text`` is the notification body;
                ``message_id`` is reused as the notification id so updates
                replace the same card.
        """
        url = f"{self.ha_url}/api/services/persistent_notification/create"
        body = {
            "title": "Xerxes",
            "message": message.text,
            "notification_id": message.message_id,
        }
        http_post(
            url,
            json_body=body,
            headers={"Authorization": f"Bearer {self.access_token}"},
            http_client=self._http,
        )
