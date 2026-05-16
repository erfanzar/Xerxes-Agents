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
"""Hindsight memory provider — knowledge graph + entity resolution."""

from __future__ import annotations

import os
from typing import Any

import httpx

from ._base import ExternalMemoryProviderBase


class HindsightProvider(ExternalMemoryProviderBase):
    """Memory provider backed by Hindsight's knowledge-graph API.

    Requires ``HINDSIGHT_API_KEY`` and ``HINDSIGHT_BANK_ID``; honours
    ``HINDSIGHT_BUDGET`` (default ``"mid"``) on search."""

    name = "hindsight"
    namespace_label = "hindsight"
    required_env = ("HINDSIGHT_API_KEY", "HINDSIGHT_BANK_ID")

    BASE_URL = "https://api.hindsight.ai/v1"

    def _client(self) -> httpx.Client:
        """Return an authenticated ``httpx.Client`` for the Hindsight API."""
        return httpx.Client(
            base_url=self.BASE_URL,
            headers={"X-Api-Key": os.environ["HINDSIGHT_API_KEY"]},
            timeout=30.0,
        )

    def _call_upstream(self, action: str, arguments: dict[str, Any]) -> Any:
        """Translate a standard action to the matching Hindsight bank endpoint."""
        bank = os.environ["HINDSIGHT_BANK_ID"]
        with self._client() as c:
            if action == "add":
                resp = c.post(f"/banks/{bank}/entries", json={"content": arguments["content"]})
            elif action == "search":
                resp = c.post(
                    f"/banks/{bank}/search",
                    json={"query": arguments["query"], "budget": os.environ.get("HINDSIGHT_BUDGET", "mid")},
                )
            elif action == "list":
                resp = c.get(f"/banks/{bank}/entries", params={"limit": arguments.get("limit", 20)})
            elif action == "remove":
                resp = c.delete(f"/banks/{bank}/entries/{arguments['entry_id']}")
            else:
                raise ValueError(f"unknown hindsight action: {action}")
            resp.raise_for_status()
            return resp.json() if resp.content else {"ok": True}


PROVIDER = HindsightProvider()

__all__ = ["PROVIDER", "HindsightProvider"]
