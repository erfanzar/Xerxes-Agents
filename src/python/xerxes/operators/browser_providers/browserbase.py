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
"""Browserbase cloud provider.

Allocates a remote chromium session through the Browserbase REST API and
returns its CDP connect URL so :class:`BrowserManager` can attach.
"""

from __future__ import annotations

import os
from typing import Any

from . import BrowserProvider, BrowserSession


class BrowserbaseProvider(BrowserProvider):
    """Provider that talks to the Browserbase REST API."""

    name = "browserbase"

    def open(self, *, headless: bool = True, **kwargs: Any) -> BrowserSession:
        """Create a remote session and return its CDP connect URL.

        Reads ``BROWSERBASE_API_KEY`` and ``BROWSERBASE_PROJECT_ID`` from
        the environment.

        Raises:
            RuntimeError: When ``httpx`` is missing or the required env
                vars are unset.
        """
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx required for Browserbase provider") from exc
        api_key = os.environ.get("BROWSERBASE_API_KEY")
        project_id = os.environ.get("BROWSERBASE_PROJECT_ID")
        if not api_key or not project_id:
            raise RuntimeError("BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID required")
        resp = httpx.post(
            "https://api.browserbase.com/v1/sessions",
            json={"projectId": project_id},
            headers={"X-BB-API-Key": api_key},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return BrowserSession(
            provider=self.name, cdp_url=data.get("connectUrl"), metadata={"session_id": data.get("id")}
        )

    def close(self, session: BrowserSession) -> None:
        """No-op: Browserbase reaps sessions after their idle timeout."""
        return None
