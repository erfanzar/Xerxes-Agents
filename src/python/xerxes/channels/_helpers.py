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
webhooks to the Xerxes ``Channel`` interface — plus simple HTTP and JSON
parsing helpers used by concrete adapters.
"""

from __future__ import annotations

import json
import logging
import typing as tp
from abc import abstractmethod

from .base import Channel, InboundHandler
from .types import ChannelMessage
from .webhooks import WebhookResponse

logger = logging.getLogger(__name__)


class WebhookChannel(Channel):
    """Abstract channel that receives messages via HTTP webhooks.

    Subclasses implement ``_parse_inbound`` and ``_send_outbound`` to
    translate platform-specific payloads to/from ``ChannelMessage``.
    """

    def __init__(self) -> None:
        """Initialize the channel with no active inbound handler."""
        self._handler: InboundHandler | None = None

    async def start(self, on_inbound: InboundHandler) -> None:
        """Register the inbound message handler.

        Args:
            on_inbound (InboundHandler): IN: async callback that will receive
                parsed ``ChannelMessage`` instances. OUT: stored internally and
                invoked for each inbound webhook.
        """
        self._handler = on_inbound

    async def stop(self) -> None:
        """Clear the inbound handler, stopping message processing."""
        self._handler = None

    async def send(self, message: ChannelMessage) -> None:
        """Send an outbound message via the channel.

        Args:
            message (ChannelMessage): IN: the message to transmit. OUT: passed
                to ``_send_outbound`` for platform-specific delivery.
        """
        await self._send_outbound(message)

    async def handle_webhook(self, headers: dict[str, str], body: bytes) -> WebhookResponse:
        """Process an incoming HTTP webhook payload.

        Args:
            headers (dict[str, str]): IN: HTTP headers from the webhook
                request. OUT: forwarded to ``_parse_inbound``.
            body (bytes): IN: raw request body. OUT: parsed by
                ``_parse_inbound`` into ``ChannelMessage`` objects.

        Returns:
            WebhookResponse: OUT: HTTP response to return to the caller.
                503 if the channel is not started, 400 on parse failure,
                200 on success.
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
        """Parse a webhook payload into ``ChannelMessage`` instances.

        Args:
            headers (dict[str, str]): IN: HTTP headers from the request.
            body (bytes): IN: raw request body.

        Returns:
            list[ChannelMessage]: OUT: parsed messages to deliver inbound.
        """

    @abstractmethod
    async def _send_outbound(self, message: ChannelMessage) -> None:
        """Transmit an outbound message to the platform.

        Args:
            message (ChannelMessage): IN: message to send.
        """


def parse_json_body(body: bytes) -> dict[str, tp.Any]:
    """Safely parse a bytes payload as JSON.

    Args:
        body (bytes): IN: raw HTTP body, possibly empty.

    Returns:
        dict[str, Any]: OUT: parsed JSON object, or an empty dict if the body
        is empty or not valid JSON / not a dict.
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
    """POST JSON data to a URL.

    Args:
        url (str): IN: target URL.
        json_body (dict[str, Any] | None): IN: JSON-serializable request body.
            OUT: serialized and sent as the request payload.
        headers (dict[str, str] | None): IN: extra HTTP headers. OUT: merged
            into the outgoing request.
        http_client (Any | None): IN: optional callable with a ``requests``-
            like signature ``(url, json=..., headers=...)``. OUT: used instead
            of ``httpx`` when provided.
        timeout (float): IN: request timeout in seconds. Defaults to 15.0.

    Returns:
        dict[str, Any]: OUT: parsed JSON response, or ``{"raw": <text>}`` if
        the response is not valid JSON.

    Raises:
        RuntimeError: If ``httpx`` is required but not installed.
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


__all__ = ["WebhookChannel", "http_post", "parse_json_body"]
