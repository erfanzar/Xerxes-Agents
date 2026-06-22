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
"""BlueBubbles channel adapter.

Talks to a self-hosted BlueBubbles server (macOS-side iMessage relay).
Parses inbound webhook payloads and sends iMessage replies through the
``/api/v1/message/text`` endpoint using the configured server password.
"""

from __future__ import annotations

import logging
import re
import typing as tp
import urllib.parse

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection

logger = logging.getLogger(__name__)


class _BlueBubblesPasswordRedactor(logging.Filter):
    """Redact BlueBubbles password query parameters from log records."""

    _PASSWORD_RE = re.compile(r"password=[^&\s]+")

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if "password=" in msg and self._PASSWORD_RE.search(msg):
            record.msg = self._PASSWORD_RE.sub("password=[REDACTED]", msg)
            record.args = ()
        return True


# Install on likely loggers so URLs with passwords don't leak in debug output.
for _name in ("httpx", "httpcore", "xerxes.channels"):
    _logger = logging.getLogger(_name)
    if not getattr(_logger, "_xerxes_bluebubbles_redactor_installed", False):
        _logger.addFilter(_BlueBubblesPasswordRedactor())
        _logger._xerxes_bluebubbles_redactor_installed = True  # type: ignore[attr-defined]


class BlueBubblesChannel(WebhookChannel):
    """BlueBubbles iMessage-bridge adapter."""

    name = "bluebubbles"

    def __init__(
        self,
        server_url: str,
        password: str,
        *,
        http_client: tp.Any = None,
    ) -> None:
        """Build the channel.

        Args:
            server_url: BlueBubbles server base URL; trailing ``/`` stripped.
            password: Server API password (sent as a URL query parameter
                on outbound calls — treat as a shared secret).
            http_client: Optional HTTP client override forwarded to
                ``http_post``.
        """
        super().__init__()
        self.server_url = server_url.rstrip("/")
        self.password = password
        self._http = http_client

    def _parse_inbound(self, headers, body):
        """Translate a BlueBubbles webhook into ``ChannelMessage``.

        Tolerates payloads with or without a top-level ``data`` wrapper,
        and either a ``chats`` list or a single ``chat`` dict. Drops
        messages with empty text (e.g. attachment-only iMessages we cannot
        yet handle).

        Args:
            headers: HTTP headers (unused).
            body: Raw JSON webhook body.

        Returns:
            One parsed inbound message, or empty when there is no body text.
        """
        data = parse_json_body(body)
        if not data:
            return []
        msg_data = data.get("data", data)
        text = msg_data.get("text") or msg_data.get("body") or ""
        chat = msg_data.get("chats", [{}])[0] if isinstance(msg_data.get("chats"), list) else msg_data.get("chat", {})
        handle = msg_data.get("handle") or {}
        if not text:
            return []
        return [
            ChannelMessage(
                text=text,
                channel=self.name,
                channel_user_id=handle.get("address", ""),
                room_id=chat.get("guid", ""),
                platform_message_id=msg_data.get("guid", ""),
                direction=MessageDirection.INBOUND,
            )
        ]

    async def _send_outbound(self, message):
        """Send one text iMessage through the BlueBubbles ``message/text`` endpoint.

        Uses ``method=private-api`` so the BlueBubbles server routes through
        the macOS private-message API rather than AppleScript fallbacks.

        Args:
            message: Outbound message. ``room_id`` is the iMessage chat GUID
                and ``text`` the body.
        """
        params = {"password": self.password}
        url = f"{self.server_url}/api/v1/message/text?{urllib.parse.urlencode(params)}"
        body = {"chatGuid": message.room_id, "message": message.text, "method": "private-api"}
        http_post(url, json_body=body, http_client=self._http)
