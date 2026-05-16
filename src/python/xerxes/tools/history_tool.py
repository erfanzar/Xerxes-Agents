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
"""History search tool for querying session and agent interaction history.

This module provides a tool for searching through historical agent interactions,
enabling agents to recall and build upon previous work.

Example:
    >>> from xerxes.tools.history_tool import SearchHistoryTool
    >>> search = SearchHistoryTool(store=my_store)
    >>> results = search("How did we implement authentication?")
"""

from __future__ import annotations

import typing as tp

if tp.TYPE_CHECKING:
    from ..session.index import SearchHit, SessionIndex
    from ..session.store import SessionStore


class SearchHistoryTool:
    """Search through historical agent interactions and session data.

    This tool provides semantic search capabilities over past agent conversations,
    enabling agents to recall relevant context from previous sessions. It can work
    with either a session store or a search index.

    Attributes:
        name: Tool identifier, always "search_history".
        store: Optional SessionStore instance.
        index: Optional SessionIndex instance.
        default_k: Default number of results to return.

    Example:
        >>> tool = SearchHistoryTool(index=my_index, default_k=10)
        >>> results = tool("previous authentication implementation")
        >>> for hit in results["hits"]:
        ...     print(hit["prompt"])
    """

    name: str = "search_history"

    def __init__(
        self,
        *,
        store: SessionStore | None = None,
        index: SessionIndex | None = None,
        default_k: int = 5,
    ) -> None:
        """Initialize the search history tool.

        Args:
            store: Optional session store for search operations.
            index: Optional search index for semantic search.
            default_k: Default maximum number of results. Defaults to 5.

        Raises:
            ValueError: If neither store nor index is provided.
        """
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
        """Search historical interactions matching the query.

        Args:
            query: Search query string to match against historical prompts.
            limit: Maximum number of results to return. Defaults to default_k.
            agent_id: Filter results to specific agent ID.
            session_id: Filter results to specific session ID.

        Returns:
            Dictionary containing:
            - query: The search query used
            - count: Number of matching results
            - hits: List of matching history entries with session_id, turn_id,
              agent_id, prompt, response, score, and timestamp

        Example:
            >>> tool = SearchHistoryTool(store=store)
            >>> results = tool("database schema", limit=3)
            >>> print(f"Found {results['count']} matches")
        """
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
