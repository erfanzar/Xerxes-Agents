# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
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
"""Shared helpers for channel adapters.

Mostly an abstract :class:`WebhookChannel` mixin: most messaging
platforms expose an HTTP POST endpoint plus an outbound REST call. The
mixin gives adapters a uniform structure.
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
    """Channel base class for platforms that ship inbound via HTTP POST.

    Subclasses implement two methods:

    - :meth:`_parse_inbound`: turn the raw POST body into one or more
      :class:`ChannelMessage`.
    - :meth:`_send_outbound`: deliver an outbound :class:`ChannelMessage`
      to the platform.

    A :class:`WebhookDispatcher` is responsible for actually routing
    HTTP requests to :meth:`handle_webhook`; the channel itself does not
    bind a port.
    """

    def __init__(self) -> None:
        """Initialise with no inbound handler attached."""
        self._handler: InboundHandler | None = None

    async def start(self, on_inbound: InboundHandler) -> None:
        """Attach the inbound handler that will receive parsed messages.

        Args:
            on_inbound: Coroutine invoked once per :class:`ChannelMessage`
                parsed from a webhook POST.
        """
        self._handler = on_inbound

    async def stop(self) -> None:
        """Detach the inbound handler so further webhooks are rejected."""
        self._handler = None

    async def send(self, message: ChannelMessage) -> None:
        """Deliver ``message`` by invoking :meth:`_send_outbound`."""
        await self._send_outbound(message)

    async def handle_webhook(self, headers: dict[str, str], body: bytes) -> WebhookResponse:
        """Parse the body and forward each :class:`ChannelMessage`."""
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
        """Convert the raw body to channel messages."""

    @abstractmethod
    async def _send_outbound(self, message: ChannelMessage) -> None:
        """Deliver an outbound message via the platform's API."""


def parse_json_body(body: bytes) -> dict[str, tp.Any]:
    """Parse a JSON request body into a dict, returning ``{}`` on error."""
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
    """Convenience POST that prefers an injected ``http_client`` callable.

    The injected client should accept ``(url, json=..., headers=...)``
    and return a dict. When absent, falls back to ``httpx``.
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
