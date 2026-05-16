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
"""Helper utilities for webhook-based channel implementations.

Provides ``WebhookChannel`` — a base class that bridges inbound HTTP
webhooks to the Xerxes ``Channel`` interface — plus shared HTTP and JSON
parsing helpers used by every concrete adapter in this package.

Also installs a logging filter that scrubs Telegram bot tokens from log
records before they leave the process. Telegram's Bot API embeds the
token directly in the URL path (e.g. ``/bot<token>/sendMessage``) and
the token cannot be moved to a header, so the only way to keep it out
of third-party log sinks is post-hoc redaction.
"""

from __future__ import annotations

import json
import logging
import re
import typing as tp
from abc import abstractmethod

from .base import Channel, InboundHandler
from .types import ChannelMessage
from .webhooks import WebhookResponse

logger = logging.getLogger(__name__)


# Redact bot tokens that show up in URLs logged by httpx or our own helpers.
# Telegram's Bot API embeds the token in the path (/bot<TOKEN>/method) and we
# can't move it to a header. The next best thing is to scrub log records so
# that a stack trace shipped to a third-party log sink doesn't leak the token.
_TELEGRAM_TOKEN_RE = re.compile(r"/bot[0-9]+:[A-Za-z0-9_-]+")


class _TelegramTokenRedactor(logging.Filter):
    """Logging filter that strips Telegram bot tokens from log records.

    Matches the ``/bot<number>:<base62>`` segment that appears in every
    Telegram Bot API URL and substitutes a ``/bot[REDACTED]`` placeholder
    on the record before downstream handlers format it. The filter always
    returns ``True`` so it never drops records — it only mutates them.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact bot tokens from ``record`` in-place.

        Args:
            record: The log record about to be emitted.

        Returns:
            Always ``True`` so the record continues through the handler chain.
        """
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if "/bot" in msg and _TELEGRAM_TOKEN_RE.search(msg):
            record.msg = _TELEGRAM_TOKEN_RE.sub("/bot[REDACTED]", msg)
            record.args = ()
        return True


def install_log_redaction() -> None:
    """Attach the Telegram-token redactor to the loggers that emit URLs.

    Targets the ``xerxes.channels``, ``httpx``, and ``httpcore`` loggers —
    httpx/httpcore log full request URLs (including bot tokens) at DEBUG.
    Idempotent: a sentinel attribute on each logger prevents duplicate
    filters when called repeatedly (e.g. once per gateway init).
    """
    filt = _TelegramTokenRedactor()
    for name in ("xerxes.channels", "httpx", "httpcore"):
        target = logging.getLogger(name)
        if getattr(target, "_xerxes_token_redactor_installed", False):
            continue
        target.addFilter(filt)
        target._xerxes_token_redactor_installed = True  # type: ignore[attr-defined]


class WebhookChannel(Channel):
    """Abstract channel that receives messages via HTTP webhooks.

    Concrete adapters override ``_parse_inbound`` to translate raw HTTP
    payloads into ``ChannelMessage`` instances and ``_send_outbound`` to
    deliver replies through the platform's API. ``handle_webhook`` does the
    common dispatch work — parse, fan out to the registered inbound handler,
    and convert exceptions into HTTP responses — so subclasses never have to
    deal with that scaffolding.
    """

    def __init__(self) -> None:
        """Initialize with no inbound handler registered yet."""
        self._handler: InboundHandler | None = None

    async def start(self, on_inbound: InboundHandler) -> None:
        """Register the inbound message handler.

        Args:
            on_inbound: Async callback invoked for every parsed
                ``ChannelMessage``. Replaces any previously registered handler.
        """
        self._handler = on_inbound

    async def stop(self) -> None:
        """Drop the inbound handler so further webhooks return 503."""
        self._handler = None

    async def send(self, message: ChannelMessage) -> None:
        """Deliver an outbound message through the platform-specific transport.

        Args:
            message: Message to send; forwarded verbatim to ``_send_outbound``.
        """
        await self._send_outbound(message)

    async def handle_webhook(self, headers: dict[str, str], body: bytes) -> WebhookResponse:
        """Process an incoming HTTP webhook payload.

        Parses the payload, dispatches each resulting ``ChannelMessage`` to
        the registered handler, and swallows handler exceptions (logged at
        WARNING) so a single bad message does not block the rest.

        Args:
            headers: HTTP headers from the webhook request.
            body: Raw request body.

        Returns:
            503 when ``start`` has not been called, 400 when ``_parse_inbound``
            raises, otherwise 200.
        """
        if self._handler is None:
            return WebhookResponse(status=503, body="channel not started")
        try:
            messages = self._parse_inbound(headers, body)
        except Exception:
            logger.warning("%s failed to parse inbound", self.name, exc_info=True)
            return WebhookResponse(status=400, body="invalid payload")
        for msg in messages:
            try:
                await self._handler(msg)
            except Exception:
                logger.warning("%s inbound handler raised", self.name, exc_info=True)
        return WebhookResponse(status=200, body="ok")

    @abstractmethod
    def _parse_inbound(self, headers: dict[str, str], body: bytes) -> list[ChannelMessage]:
        """Convert a raw webhook payload into ``ChannelMessage`` instances.

        Implementations should return an empty list (never raise) for events
        that should be ignored — URL verifications, bot-loopback echoes,
        unsupported event types, signature mismatches, and so on.

        Args:
            headers: HTTP headers from the webhook request.
            body: Raw request body.

        Returns:
            Parsed inbound messages, or an empty list when the payload should
            be silently dropped.
        """

    @abstractmethod
    async def _send_outbound(self, message: ChannelMessage) -> None:
        """Deliver one message to the upstream messaging platform.

        Args:
            message: Outbound message; subclasses decide which fields
                (``room_id``, ``channel_user_id``, ``reply_to``, ...) map to
                the platform's recipient and threading semantics.
        """


def parse_json_body(body: bytes) -> dict[str, tp.Any]:
    """Decode a bytes payload as a JSON object, never raising.

    Args:
        body: Raw HTTP body. May be empty.

    Returns:
        The decoded mapping, or an empty dict when the body is empty,
        malformed, or decodes to something that is not a JSON object.
    """
    if not body:
        return {}
    try:
        data = json.loads(body)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def http_post(
    url: str,
    *,
    json_body: dict[str, tp.Any] | None = None,
    headers: dict[str, str] | None = None,
    http_client: tp.Any | None = None,
    timeout: float = 15.0,
) -> dict[str, tp.Any]:
    """POST JSON to a URL and return the decoded response.

    When ``http_client`` is supplied the caller controls transport entirely
    (used by tests and by adapters that want connection pooling); otherwise
    ``httpx`` performs a synchronous POST. Non-2xx responses raise via
    ``httpx`` — adapters relying on the default path should expect
    ``HTTPStatusError``.

    Args:
        url: Target URL.
        json_body: JSON-serialisable request body.
        headers: Extra HTTP headers merged into the outgoing request.
        http_client: Optional ``requests``-like callable with signature
            ``(url, json=..., headers=...)``. Returning a dict short-circuits
            JSON parsing; otherwise the return value is treated as a JSON
            string.
        timeout: Request timeout in seconds. Defaults to 15.

    Returns:
        Parsed JSON response, or ``{"raw": <text>}`` if the body is not
        valid JSON.

    Raises:
        RuntimeError: When ``httpx`` is needed but not installed.
    """
    if http_client is not None:
        out = http_client(url, json=json_body, headers=headers)
        if isinstance(out, dict):
            return out
        try:
            return json.loads(out)
        except Exception:
            return {"raw": out}
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx required for channel HTTP calls") from exc
    resp = httpx.post(url, json=json_body, headers=headers or {}, timeout=timeout)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text}


__all__ = ["WebhookChannel", "http_post", "install_log_redaction", "parse_json_body"]
