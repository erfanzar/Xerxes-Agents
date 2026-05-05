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

Provides a lightweight registry that maps channel names to async webhook
handlers and returns standardized ``WebhookResponse`` objects.
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
        status: HTTP status code.
        body: Response body string.
        headers: Optional response headers.
    """

    status: int = 200
    body: str = ""
    headers: dict[str, str] | None = None


class WebhookDispatcher:
    """Routes incoming webhook requests to the appropriate handler."""

    def __init__(self) -> None:
        """Initialize an empty dispatcher."""
        self._handlers: dict[str, WebhookHandler] = {}

    def register(self, name: str, handler: WebhookHandler) -> None:
        """Register a webhook handler for a channel.

        Args:
            name (str): IN: channel identifier.
            handler (WebhookHandler): IN: async callable that processes
                webhook payloads for this channel.
        """
        self._handlers[name] = handler

    def unregister(self, name: str) -> None:
        """Remove a registered webhook handler.

        Args:
            name (str): IN: channel identifier to remove.
        """
        self._handlers.pop(name, None)

    def names(self) -> list[str]:
        """Return all registered handler names.

        Returns:
            list[str]: OUT: list of channel identifiers.
        """
        return list(self._handlers.keys())

    async def dispatch(self, name: str, headers: dict[str, str], body: bytes) -> WebhookResponse:
        """Dispatch a webhook request to the registered handler.

        Args:
            name (str): IN: channel identifier.
            headers (dict[str, str]): IN: HTTP request headers.
            body (bytes): IN: raw request body.

        Returns:
            WebhookResponse: OUT: 404 if no handler is found, 500 on handler
            exception, otherwise the handler's response.
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
