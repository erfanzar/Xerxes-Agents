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
"""Pluggable browser providers used by :class:`BrowserManager`.

Each provider is an instance of :class:`BrowserProvider` that knows how to
open a session and return either an in-process Playwright handle (for the
``local`` provider) or a CDP/WebSocket URL the manager can attach to (for
remote providers such as ``camofox``, ``browserbase``, ``browser_use``,
``firecrawl``).

Production wiring is deliberately thin: heavyweight imports happen only
when a provider's ``open`` method is invoked, so the default Xerxes path
with ``provider="local"`` works with just Playwright installed. Built-in
providers register themselves into the module-level registry at import.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

SUPPORTED_PROVIDERS: tuple[str, ...] = ("local", "camofox", "browserbase", "browser_use", "firecrawl")


@dataclass
class BrowserSession:
    """Result of :meth:`BrowserProvider.open`.

    Attributes:
        provider: Name of the provider that produced this session.
        cdp_url: WebSocket URL when the session is remote/CDP-based;
            ``None`` for in-process providers.
        page: Playwright ``Page`` (or equivalent) when the provider runs
            in-process; ``None`` for remote providers.
        metadata: Free-form provider-specific bookkeeping such as a cloud
            session id, billing handle, or proxy descriptor.
    """

    provider: str
    cdp_url: str | None = None
    page: Any | None = None
    metadata: dict[str, Any] | None = None


class BrowserProvider(ABC):
    """Abstract base for every browser provider implementation.

    Subclasses must set :attr:`name` to the registry key and implement
    :meth:`open` / :meth:`close`. Providers should keep constructors cheap
    and defer heavyweight imports until :meth:`open` is called so the
    registry stays usable in minimal environments.
    """

    name: str = ""

    @abstractmethod
    def open(self, *, headless: bool = True, **kwargs: Any) -> BrowserSession:
        """Start a browser session and return its :class:`BrowserSession`."""

    @abstractmethod
    def close(self, session: BrowserSession) -> None:
        """Release any resources held by ``session``."""


_REGISTRY: dict[str, BrowserProvider] = {}


def register(provider: BrowserProvider) -> None:
    """Register ``provider`` under its :attr:`BrowserProvider.name`."""
    _REGISTRY[provider.name] = provider


def registry() -> dict[str, BrowserProvider]:
    """Return a shallow copy of the provider registry."""
    return dict(_REGISTRY)


def get(name: str) -> BrowserProvider | None:
    """Look up a provider by name, returning ``None`` if not registered."""
    return _REGISTRY.get(name)


# Eager registration of the built-ins (light constructors only — they
# import heavy libs lazily when ``open()`` is called).
from .browser_use import BrowserUseProvider
from .browserbase import BrowserbaseProvider
from .camofox import CamofoxProvider
from .firecrawl import FirecrawlProvider
from .local import LocalProvider

for _provider in (LocalProvider(), CamofoxProvider(), BrowserbaseProvider(), BrowserUseProvider(), FirecrawlProvider()):
    register(_provider)


__all__ = ["SUPPORTED_PROVIDERS", "BrowserProvider", "BrowserSession", "get", "register", "registry"]
