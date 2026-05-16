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
"""Cross-platform ``send_message`` dispatcher.

Routes outbound messages to any platform whose adapter has registered a
:data:`SendFn` callback. The default registry lives in-process; tests and
adapters inject themselves via :func:`register_platform`, so the tool can run
without real Telegram, Discord, or Slack credentials.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

SendFn = Callable[[str, str, dict[str, Any]], dict[str, Any]]


class _DefaultRegistry:
    """In-memory channel registry backing :func:`send_message`.

    Platform names are folded to lower-case so callers don't need to match
    the adapter's exact casing.
    """

    def __init__(self) -> None:
        """Initialize with an empty handler table."""
        self._handlers: dict[str, SendFn] = {}

    def register(self, platform: str, fn: SendFn) -> None:
        """Register ``fn`` as the send-callback for ``platform`` (case-insensitive)."""
        self._handlers[platform.lower()] = fn

    def send(self, platform: str, recipient: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Dispatch ``payload`` to the handler registered for ``platform``."""
        fn = self._handlers.get(platform.lower())
        if fn is None:
            return {"ok": False, "error": f"unknown platform: {platform}"}
        return fn(platform, recipient, payload)


_default_registry = _DefaultRegistry()


def register_platform(platform: str, fn: SendFn) -> None:
    """Register a send-callable on the module-level default registry.

    Adapters call this at import time; the callback receives ``(platform,
    recipient, payload)`` and returns a result dict.
    """
    _default_registry.register(platform, fn)


def send_message(
    *,
    platform: str,
    recipient: str,
    text: str = "",
    files: list[str] | None = None,
    reply_to: str | None = None,
) -> dict[str, Any]:
    """Deliver a message and/or attachments through a registered platform.

    Args:
        platform: Platform identifier (e.g. ``telegram``, ``discord``); must
            match a name registered via :func:`register_platform`.
        recipient: Platform-specific identifier (chat id, user handle, etc.).
        text: Message body. May be empty if ``files`` is provided.
        files: Optional list of file paths to attach. The platform adapter
            decides how each path is uploaded.
        reply_to: Optional upstream message id to quote in the reply.

    Returns:
        Dict containing at least ``ok`` (bool) plus adapter-specific fields.
        Validation failures return ``{"ok": False, "error": ...}`` without
        invoking any adapter.
    """
    if not platform:
        return {"ok": False, "error": "platform required"}
    if not recipient:
        return {"ok": False, "error": "recipient required"}
    if not text and not files:
        return {"ok": False, "error": "text or files required"}
    payload: dict[str, Any] = {"text": text, "files": list(files or [])}
    if reply_to is not None:
        payload["reply_to"] = reply_to
    return _default_registry.send(platform, recipient, payload)


__all__ = ["SendFn", "register_platform", "send_message"]
