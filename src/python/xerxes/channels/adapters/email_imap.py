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
"""Email (SMTP) channel adapter.

Supports sending outbound email via SMTP and parsing inbound email
webhook payloads.
"""

from __future__ import annotations

import logging
import typing as tp

from .._helpers import WebhookChannel, parse_json_body
from ..types import ChannelMessage, MessageDirection

logger = logging.getLogger(__name__)


class EmailChannel(WebhookChannel):
    """Channel implementation for email via SMTP."""

    name = "email"

    def __init__(
        self,
        smtp_host: str = "localhost",
        smtp_port: int = 25,
        smtp_user: str = "",
        smtp_password: str = "",
        from_address: str = "",
        *,
        smtp_sender: tp.Callable[[str, str, str, str], None] | None = None,
    ) -> None:
        """Initialize the Email channel.

        Args:
            smtp_host (str): IN: SMTP server hostname. Defaults to "localhost".
                OUT: stored for outbound connections.
            smtp_port (int): IN: SMTP server port. Defaults to 25.
                OUT: stored for outbound connections.
            smtp_user (str): IN: SMTP username. OUT: used for authentication.
            smtp_password (str): IN: SMTP password. OUT: used for authentication.
            from_address (str): IN: sender email address. Defaults to
                ``smtp_user`` if not provided.
            smtp_sender (Callable | None): IN: optional callable
                ``(from_addr, to_addr, subject, body) -> None`` that bypasses
                the built-in smtplib logic.
        """
        super().__init__()
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.from_address = from_address or smtp_user
        self._smtp_sender = smtp_sender

    def _parse_inbound(self, headers, body):
        """Parse an email webhook payload into ``ChannelMessage``.

        Args:
            headers (dict[str, str]): IN: HTTP headers (unused).
            body (bytes): IN: raw JSON webhook body.

        Returns:
            list[ChannelMessage]: OUT: parsed inbound messages.
        """
        data = parse_json_body(body)
        if not data:
            return []
        return [
            ChannelMessage(
                text=data.get("text", "") or data.get("html", ""),
                channel=self.name,
                channel_user_id=data.get("from", ""),
                room_id=data.get("to", ""),
                platform_message_id=data.get("message_id", ""),
                direction=MessageDirection.INBOUND,
                metadata={"subject": data.get("subject", "")},
            )
        ]

    async def _send_outbound(self, message):
        """Send an email message via SMTP.

        Args:
            message (ChannelMessage): IN: message to send. ``room_id`` or
                ``channel_user_id`` is used as the recipient, ``text`` as the
                body, and ``metadata["subject"]`` as the subject.

        Raises:
            ValueError: If neither ``room_id`` nor ``channel_user_id`` is set.
        """
        to_addr = message.room_id or message.channel_user_id
        if not to_addr:
            raise ValueError("EmailChannel.send requires room_id or channel_user_id (recipient)")
        subject = (message.metadata or {}).get("subject", "Re:")
        if self._smtp_sender is not None:
            self._smtp_sender(self.from_address, to_addr, subject, message.text)
            return
        try:
            import smtplib
            from email.mime.text import MIMEText
        except ImportError:
            logger.warning("smtplib unavailable; cannot send email")
            return
        msg = MIMEText(message.text, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = self.from_address
        msg["To"] = to_addr
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as smtp:
            if self.smtp_user:
                try:
                    smtp.starttls()
                except Exception:
                    pass
                smtp.login(self.smtp_user, self.smtp_password)
            smtp.send_message(msg)
