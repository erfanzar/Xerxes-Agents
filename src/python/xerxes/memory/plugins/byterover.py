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
"""ByteRover memory provider — hierarchical context tree."""

from __future__ import annotations

import os
from typing import Any

import httpx

from ._base import ExternalMemoryProviderBase


class ByteRoverProvider(ExternalMemoryProviderBase):
    """Memory provider backed by ByteRover's hierarchical context tree API.

    Requires ``BRV_API_KEY``. The ``add`` action accepts an optional
    ``parent`` argument to attach the new node beneath an existing one."""

    name = "byterover"
    namespace_label = "brv"
    required_env = ("BRV_API_KEY",)

    BASE_URL = "https://api.byterover.dev/v1"

    def _client(self) -> httpx.Client:
        """Return an authenticated ``httpx.Client`` for the ByteRover API."""
        return httpx.Client(
            base_url=self.BASE_URL,
            headers={"Authorization": f"Bearer {os.environ['BRV_API_KEY']}"},
            timeout=30.0,
        )

    def _call_upstream(self, action: str, arguments: dict[str, Any]) -> Any:
        """Translate a standard action to the matching ByteRover ``nodes`` endpoint."""
        with self._client() as c:
            if action == "add":
                resp = c.post("/nodes", json={"content": arguments["content"], "parent": arguments.get("parent")})
            elif action == "search":
                resp = c.post("/nodes/search", json={"query": arguments["query"]})
            elif action == "list":
                resp = c.get("/nodes", params={"limit": arguments.get("limit", 20)})
            elif action == "remove":
                resp = c.delete(f"/nodes/{arguments['entry_id']}")
            else:
                raise ValueError(f"unknown byterover action: {action}")
            resp.raise_for_status()
            return resp.json() if resp.content else {"ok": True}


PROVIDER = ByteRoverProvider()

__all__ = ["PROVIDER", "ByteRoverProvider"]
