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

Outbound: sends ``text/plain`` messages via ``smtplib`` (with optional
STARTTLS+AUTH when credentials are supplied) or through an injected
``smtp_sender`` callable for tests. Inbound: parses an existing
mail-to-webhook bridge's JSON payload — the module does *not* run an
IMAP poller, despite the filename. Subjects come in via
``metadata['subject']``.
"""

from __future__ import annotations

import logging
import typing as tp

from .._helpers import WebhookChannel, parse_json_body
from ..types import ChannelMessage, MessageDirection

logger = logging.getLogger(__name__)


class EmailChannel(WebhookChannel):
    """SMTP-out / webhook-in email adapter."""

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
        """Build the channel.

        Args:
            smtp_host: SMTP server hostname. Defaults to ``"localhost"``.
            smtp_port: SMTP server port. Defaults to 25.
            smtp_user: SMTP username. Triggers STARTTLS+AUTH on send.
            smtp_password: SMTP password.
            from_address: Envelope ``From`` and ``From:`` header. Falls
                back to ``smtp_user`` when empty.
            smtp_sender: Optional callable
                ``(from_addr, to_addr, subject, body) → None`` that
                bypasses ``smtplib``. Used by tests and by deployments
                that want to plug in a custom mailer.
        """
        super().__init__()
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.from_address = from_address or smtp_user
        self._smtp_sender = smtp_sender

    def _parse_inbound(self, headers, body):
        """Translate an email-bridge webhook payload into ``ChannelMessage``.

        Expects a JSON shape like ``{"from": ..., "to": ..., "subject": ...,
        "text": ..., "html": ..., "message_id": ...}`` produced by services
        such as Mailgun routes or Postmark inbound. Prefers ``text``; falls
        back to ``html`` when only an HTML body is provided.

        Args:
            headers: HTTP headers (unused).
            body: Raw JSON webhook body.

        Returns:
            One parsed message, or empty when the payload is empty.
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
        """Send one email — via ``smtp_sender`` when set, otherwise ``smtplib``.

        When ``smtp_user`` is non-empty the SMTP path attempts STARTTLS
        (best-effort, ignored on failure) and authenticates before
        ``send_message``. If ``smtplib`` is unavailable the call logs a
        warning and returns without sending.

        Args:
            message: Outbound message. ``room_id`` or ``channel_user_id``
                is the recipient; ``text`` is the body; subject comes from
                ``metadata['subject']`` (defaults to ``"Re:"``).

        Raises:
            ValueError: Neither ``room_id`` nor ``channel_user_id`` is set.
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
