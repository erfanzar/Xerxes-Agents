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
"""Webhook dispatcher for routing HTTP callbacks to channel handlers.

The bridge server exposes one HTTP endpoint per channel and forwards each
request into ``WebhookDispatcher.dispatch``; the dispatcher looks up the
matching ``WebhookHandler`` and returns a ``WebhookResponse`` the HTTP
layer can turn into a response. Handler exceptions are caught and surfaced
as HTTP 500 so a buggy adapter cannot crash the bridge.
"""

from __future__ import annotations

import logging
import typing as tp
from dataclasses import dataclass

logger = logging.getLogger(__name__)
WebhookHandler = tp.Callable[[dict[str, str], bytes], tp.Awaitable["WebhookResponse"]]


@dataclass
class WebhookResponse:
    """HTTP response returned by a webhook handler.

    Attributes:
        status: HTTP status code; defaults to 200.
        body: Response body. The HTTP layer is responsible for encoding.
        headers: Extra response headers, or ``None`` for none.
    """

    status: int = 200
    body: str = ""
    headers: dict[str, str] | None = None


class WebhookDispatcher:
    """Maps channel names to async webhook handlers.

    A thin wrapper around a ``dict[name, handler]`` plus error containment:
    unknown names return 404, and handler exceptions are caught, logged at
    WARNING, and surfaced as 500 so a single bad adapter cannot tear down
    the HTTP server.
    """

    def __init__(self) -> None:
        """Build a dispatcher with no handlers registered."""
        self._handlers: dict[str, WebhookHandler] = {}

    def register(self, name: str, handler: WebhookHandler) -> None:
        """Register or replace the handler for ``name``.

        Args:
            name: Channel identifier (typically matches ``Channel.name``).
            handler: Async callable that processes one webhook payload.
        """
        self._handlers[name] = handler

    def unregister(self, name: str) -> None:
        """Remove the handler for ``name``; no-op if missing.

        Args:
            name: Channel identifier passed to ``register``.
        """
        self._handlers.pop(name, None)

    def names(self) -> list[str]:
        """Return all registered channel identifiers in insertion order."""
        return list(self._handlers.keys())

    async def dispatch(self, name: str, headers: dict[str, str], body: bytes) -> WebhookResponse:
        """Look up and invoke the handler for ``name``.

        Args:
            name: Channel identifier.
            headers: HTTP request headers.
            body: Raw request body.

        Returns:
            404 ``WebhookResponse`` when no handler is registered, 500 when
            the handler raises (logged at WARNING), otherwise whatever the
            handler returned.
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
