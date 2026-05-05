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
"""History tool module for Xerxes.

Exports:
    - SearchHistoryTool"""

from __future__ import annotations

import typing as tp

if tp.TYPE_CHECKING:
    from ..session.index import SearchHit, SessionIndex
    from ..session.store import SessionStore


class SearchHistoryTool:
    """Search history tool.

    Attributes:
        name (str): name."""

    name: str = "search_history"

    def __init__(
        self,
        *,
        store: SessionStore | None = None,
        index: SessionIndex | None = None,
        default_k: int = 5,
    ) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            store (SessionStore | None, optional): IN: store. Defaults to None. OUT: Consumed during execution.
            index (SessionIndex | None, optional): IN: index. Defaults to None. OUT: Consumed during execution.
            default_k (int, optional): IN: default k. Defaults to 5. OUT: Consumed during execution."""

        if store is None and index is None:
            raise ValueError("SearchHistoryTool requires a store or an index")
        self.store = store
        self.index = index
        self.default_k = default_k

    def __call__(
        self,
        query: str,
        limit: int | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, tp.Any]:
        """Dunder method for call.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            query (str): IN: query. OUT: Consumed during execution.
            limit (int | None, optional): IN: limit. Defaults to None. OUT: Consumed during execution.
            agent_id (str | None, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
            session_id (str | None, optional): IN: session id. Defaults to None. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        k = limit or self.default_k
        if self.index is not None:
            hits: list[SearchHit] = self.index.search(query, k=k, agent_id=agent_id, session_id=session_id)
        else:
            assert self.store is not None
            hits = self.store.search(query, k=k, agent_id=agent_id, session_id=session_id)
        return {
            "query": query,
            "count": len(hits),
            "hits": [
                {
                    "session_id": h.session_id,
                    "turn_id": h.turn_id,
                    "agent_id": h.agent_id,
                    "prompt": h.prompt,
                    "response": h.response,
                    "score": round(h.score, 4),
                    "timestamp": h.timestamp,
                }
                for h in hits
            ],
        }


__all__ = ["SearchHistoryTool"]
