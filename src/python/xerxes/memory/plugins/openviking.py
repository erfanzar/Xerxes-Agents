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
"""OpenViking (Volcengine context DB) memory provider."""

from __future__ import annotations

import os
from typing import Any

import httpx

from ._base import ExternalMemoryProviderBase


class OpenVikingProvider(ExternalMemoryProviderBase):
    """Memory provider backed by Volcengine's OpenViking context API.

    Requires both ``OPENVIKING_ENDPOINT`` and ``OPENVIKING_API_KEY``."""

    name = "openviking"
    namespace_label = "viking"
    required_env = ("OPENVIKING_ENDPOINT", "OPENVIKING_API_KEY")

    def _client(self) -> httpx.Client:
        """Return an authenticated ``httpx.Client`` targeting ``OPENVIKING_ENDPOINT``."""
        return httpx.Client(
            base_url=os.environ["OPENVIKING_ENDPOINT"],
            headers={"Authorization": f"Bearer {os.environ['OPENVIKING_API_KEY']}"},
            timeout=30.0,
        )

    def _call_upstream(self, action: str, arguments: dict[str, Any]) -> Any:
        """Translate a standard action to the matching OpenViking endpoint."""
        with self._client() as c:
            if action == "add":
                resp = c.post("/v1/contexts", json={"content": arguments["content"]})
            elif action == "search":
                resp = c.post("/v1/contexts/search", json={"query": arguments["query"]})
            elif action == "list":
                resp = c.get("/v1/contexts", params={"limit": arguments.get("limit", 20)})
            elif action == "remove":
                resp = c.delete(f"/v1/contexts/{arguments['entry_id']}")
            else:
                raise ValueError(f"unknown openviking action: {action}")
            resp.raise_for_status()
            return resp.json() if resp.content else {"ok": True}


PROVIDER = OpenVikingProvider()

__all__ = ["PROVIDER", "OpenVikingProvider"]
