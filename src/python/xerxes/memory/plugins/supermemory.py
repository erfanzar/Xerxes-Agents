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
"""Supermemory cloud memory provider (https://supermemory.ai)."""

from __future__ import annotations

import os
from typing import Any

import httpx

from ._base import ExternalMemoryProviderBase


class SupermemoryProvider(ExternalMemoryProviderBase):
    """Memory provider backed by supermemory.ai's HTTP API.

    Requires ``SUPERMEMORY_API_KEY``."""

    name = "supermemory"
    namespace_label = "super"
    required_env = ("SUPERMEMORY_API_KEY",)

    BASE_URL = "https://api.supermemory.ai/v1"

    def _client(self) -> httpx.Client:
        """Return an authenticated ``httpx.Client`` for the Supermemory API."""
        return httpx.Client(
            base_url=self.BASE_URL,
            headers={"Authorization": f"Bearer {os.environ['SUPERMEMORY_API_KEY']}"},
            timeout=30.0,
        )

    def _call_upstream(self, action: str, arguments: dict[str, Any]) -> Any:
        """Translate a standard action to the matching Supermemory endpoint."""
        with self._client() as c:
            if action == "add":
                resp = c.post("/memories", json={"content": arguments["content"]})
            elif action == "search":
                resp = c.post("/memories/search", json={"q": arguments["query"]})
            elif action == "list":
                resp = c.get("/memories", params={"limit": arguments.get("limit", 20)})
            elif action == "remove":
                resp = c.delete(f"/memories/{arguments['entry_id']}")
            else:
                raise ValueError(f"unknown supermemory action: {action}")
            resp.raise_for_status()
            return resp.json() if resp.content else {"ok": True}


PROVIDER = SupermemoryProvider()

__all__ = ["PROVIDER", "SupermemoryProvider"]
