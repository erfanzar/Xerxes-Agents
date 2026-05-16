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
"""Honcho AI memory provider (https://github.com/plastic-labs/honcho).

Available when the ``honcho-ai`` SDK is installed and ``HONCHO_API_KEY``
is set."""

from __future__ import annotations

from typing import Any

from ._base import ExternalMemoryProviderBase


class HonchoProvider(ExternalMemoryProviderBase):
    """Memory provider backed by Honcho's notes API.

    Requires the ``honcho-ai`` SDK and ``HONCHO_API_KEY``."""

    name = "honcho"
    namespace_label = "honcho"
    required_module = "honcho"
    required_env = ("HONCHO_API_KEY",)

    def _call_upstream(self, action: str, arguments: dict[str, Any]) -> Any:
        """Translate a standard action to the Honcho ``notes`` client call."""
        from honcho import Honcho  # type: ignore

        client = Honcho()
        if action == "add":
            note = client.notes.create(content=arguments["content"], tags=arguments.get("tags", []))
            return {"id": getattr(note, "id", ""), "content": getattr(note, "content", "")}
        if action == "search":
            results = client.notes.search(query=arguments["query"], limit=arguments.get("limit", 10))
            return [{"id": r.id, "content": r.content} for r in results]
        if action == "list":
            results = client.notes.list(limit=arguments.get("limit", 20))
            return [{"id": r.id, "content": r.content} for r in results]
        if action == "remove":
            client.notes.delete(arguments["entry_id"])
            return {"removed": True}
        raise ValueError(f"unknown honcho action: {action}")


PROVIDER = HonchoProvider()

__all__ = ["PROVIDER", "HonchoProvider"]
