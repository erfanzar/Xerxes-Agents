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
"""Firecrawl provider — request-shaped scrape returning markdown payloads."""

from __future__ import annotations

import os
from typing import Any

from . import BrowserProvider, BrowserSession


class FirecrawlProvider(BrowserProvider):
    """Provider sentinel that routes scrape requests through Firecrawl.

    Firecrawl is request-shaped rather than session-shaped: the manager
    interprets the returned :class:`BrowserSession` as a directive to use
    Firecrawl's ``scrape()`` endpoint per ``web.open``.
    """

    name = "firecrawl"

    def open(self, *, headless: bool = True, **kwargs: Any) -> BrowserSession:
        """Validate that ``FIRECRAWL_API_KEY`` is set and return a marker.

        Raises:
            RuntimeError: When the API key env var is missing.
        """
        api_key = os.environ.get("FIRECRAWL_API_KEY")
        if not api_key:
            raise RuntimeError("FIRECRAWL_API_KEY required")
        # Firecrawl is request-shaped, not session-shaped — return a tag
        # session the BrowserManager wrapper interprets as "use scrape()".
        return BrowserSession(provider=self.name, metadata={"api_key_present": True})

    def close(self, session: BrowserSession) -> None:
        """No-op: Firecrawl has no persistent session to release."""
        return None
