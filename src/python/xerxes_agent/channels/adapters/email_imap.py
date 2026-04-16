# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""Email channel — inbound via webhook (forwarder) + outbound via SMTP.

Inbound is intentionally not IMAP-IDLE so the same adapter works in
serverless deployments. A small forwarder script (``msmtp -t`` /
SendGrid Inbound Parse / Mailgun routes) POSTs the parsed email JSON
to ``/webhooks/email`` and the channel re-emits it as a
:class:`ChannelMessage`.

Outbound uses :mod:`smtplib`; the constructor accepts an injectable
``smtp_sender`` for tests.
"""

from __future__ import annotations

import logging
import typing as tp

from .._helpers import WebhookChannel, parse_json_body
from ..types import ChannelMessage, MessageDirection

logger = logging.getLogger(__name__)


class EmailChannel(WebhookChannel):
    """Inbound: HTTP forwarder; outbound: SMTP.

    Inbound JSON shape (forwarder is responsible for producing this):
    ``{"from": "...", "to": "...", "subject": "...", "text": "...", "message_id": "..."}``.
    """

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
        """Configure SMTP send parameters and optional injectable sender.

        Args:
            smtp_host: Mail server hostname.
            smtp_port: Mail server port.
            smtp_user: SMTP auth username (empty to disable auth).
            smtp_password: SMTP auth password.
            from_address: Envelope + header ``From`` address; defaults to
                ``smtp_user`` when empty.
            smtp_sender: Optional callable ``(from, to, subject, body)``
                used instead of :mod:`smtplib` for tests.
        """
        super().__init__()
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.from_address = from_address or smtp_user
        self._smtp_sender = smtp_sender

    def _parse_inbound(self, headers, body):
        """Map the forwarder JSON (``from``/``to``/``text``/``subject``)
        to a :class:`ChannelMessage`; ``text`` falls back to ``html``."""
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
        """Send ``message.text`` via the injected sender or :mod:`smtplib`.

        Recipient is ``message.room_id`` (falling back to ``channel_user_id``);
        subject is lifted from ``metadata['subject']`` or defaults to ``"Re:"``.
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
