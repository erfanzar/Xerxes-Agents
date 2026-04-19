# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic webhook ingress for channel adapters.

A small dispatcher that adapters can register against — the daemon /
API server mounts this once and forwards every ``POST /webhooks/<name>``
request to the matching channel's ``handle_webhook(headers, body)``
coroutine. Adapters that do not need webhooks (polling adapters like
Telegram or IMAP IDLE) can simply ignore this dispatcher.
"""

from __future__ import annotations

import logging
import typing as tp
from dataclasses import dataclass

logger = logging.getLogger(__name__)
WebhookHandler = tp.Callable[[dict[str, str], bytes], tp.Awaitable["WebhookResponse"]]


@dataclass
class WebhookResponse:
    """Result of a webhook handler call.

    Attributes:
        status: HTTP status code to return to the caller.
        body: Response body (encoded to UTF-8 by the caller).
        headers: Optional response headers.
    """

    status: int = 200
    body: str = ""
    headers: dict[str, str] | None = None


class WebhookDispatcher:
    """Routes ``POST /webhooks/<name>`` to registered handlers."""

    def __init__(self) -> None:
        """Initialise the dispatcher with no handlers registered."""
        self._handlers: dict[str, WebhookHandler] = {}

    def register(self, name: str, handler: WebhookHandler) -> None:
        """Bind *handler* to webhook path ``/webhooks/<name>``."""
        self._handlers[name] = handler

    def unregister(self, name: str) -> None:
        """Remove the handler previously bound under ``name``."""
        self._handlers.pop(name, None)

    def names(self) -> list[str]:
        """Return the list of webhook paths currently bound."""
        return list(self._handlers.keys())

    async def dispatch(self, name: str, headers: dict[str, str], body: bytes) -> WebhookResponse:
        """Invoke the handler registered for *name*.

        Returns a 404 :class:`WebhookResponse` when no handler is bound.
        Exceptions from handlers are converted to ``500`` responses with
        an empty body so the request never crashes the ingress server.
        """
        handler = self._handlers.get(name)
        if handler is None:
            return WebhookResponse(status=404, body=f"unknown channel {name!r}")
        try:
            return await handler(headers, body)
        except Exception:
            logger.warning("Webhook handler %s raised", name, exc_info=True)
            return WebhookResponse(status=500, body="")


__all__ = ["WebhookDispatcher", "WebhookHandler", "WebhookResponse"]
