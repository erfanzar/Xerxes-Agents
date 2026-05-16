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
"""Mem0 cloud memory provider (https://mem0.ai)."""

from __future__ import annotations

import os
from typing import Any

from ._base import ExternalMemoryProviderBase


class Mem0Provider(ExternalMemoryProviderBase):
    """Memory provider backed by mem0.ai's HTTP API.

    Pure HTTP (no SDK dependency). Requires ``MEM0_API_KEY`` and
    optionally honours ``MEM0_USER_ID`` (defaults to ``"xerxes"``)."""

    name = "mem0"
    namespace_label = "mem0"
    required_module = None  # Pure HTTP — no SDK required.
    required_env = ("MEM0_API_KEY",)

    BASE_URL = "https://api.mem0.ai/v1"

    def _client(self):
        """Return an authenticated ``httpx.Client`` for the mem0 API."""
        import httpx

        return httpx.Client(
            base_url=self.BASE_URL,
            headers={"Authorization": f"Bearer {os.environ['MEM0_API_KEY']}"},
            timeout=30.0,
        )

    def _call_upstream(self, action: str, arguments: dict[str, Any]) -> Any:
        """Translate a standard action to the matching mem0 HTTP endpoint."""
        user_id = os.environ.get("MEM0_USER_ID", "xerxes")
        with self._client() as c:
            if action == "add":
                resp = c.post(
                    "/memories",
                    json={"messages": [{"role": "user", "content": arguments["content"]}], "user_id": user_id},
                )
                resp.raise_for_status()
                return resp.json()
            if action == "search":
                resp = c.post("/memories/search", json={"query": arguments["query"], "user_id": user_id})
                resp.raise_for_status()
                return resp.json()
            if action == "list":
                resp = c.get("/memories", params={"user_id": user_id, "limit": arguments.get("limit", 20)})
                resp.raise_for_status()
                return resp.json()
            if action == "remove":
                resp = c.delete(f"/memories/{arguments['entry_id']}")
                resp.raise_for_status()
                return {"removed": True}
        raise ValueError(f"unknown mem0 action: {action}")


PROVIDER = Mem0Provider()

__all__ = ["PROVIDER", "Mem0Provider"]
