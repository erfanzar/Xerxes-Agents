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
"""Browser Use cloud provider.

Allocates a remote chromium session via the Browser Use REST API and
returns its CDP URL for :class:`BrowserManager` to attach to.
"""

from __future__ import annotations

import os
from typing import Any

from . import BrowserProvider, BrowserSession


class BrowserUseProvider(BrowserProvider):
    """Provider that drives a remote Browser Use cloud session."""

    name = "browser_use"

    def open(self, *, headless: bool = True, **kwargs: Any) -> BrowserSession:
        """POST to ``/v1/sessions`` and return the resulting CDP descriptor.

        Reads ``BROWSER_USE_API_KEY`` from the environment.

        Raises:
            RuntimeError: When ``httpx`` is missing or the API key is unset.
        """
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx required for Browser Use provider") from exc
        api_key = os.environ.get("BROWSER_USE_API_KEY")
        if not api_key:
            raise RuntimeError("BROWSER_USE_API_KEY required")
        resp = httpx.post(
            "https://api.browser-use.com/v1/sessions",
            json={"headless": headless},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return BrowserSession(provider=self.name, cdp_url=data.get("cdp_url"), metadata={"session_id": data.get("id")})

    def close(self, session: BrowserSession) -> None:
        """No-op: remote sessions are reaped by the Browser Use service."""
        return None
