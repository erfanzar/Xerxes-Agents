# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
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
"""Agent-callable ``search_history`` tool.

Bridges the agent runtime to :class:`SessionStore.search` /
:class:`SessionIndex.search` so the agent itself can recall past
sessions when the user asks "did we discuss this before?".

Concrete tools register a :class:`SearchHistoryTool` instance bound to
the runtime's session store / index and expose its :meth:`__call__`
method to the LLM under the name ``search_history``.
"""

from __future__ import annotations

import typing as tp

if tp.TYPE_CHECKING:
    from ..session.index import SearchHit, SessionIndex
    from ..session.store import SessionStore


class SearchHistoryTool:
    """Callable wrapper that exposes ``search_history`` to the agent.

    Pass either a ``SessionStore`` (uses linear scan / overridden
    search) or a ``SessionIndex`` (preferred for cross-session FTS5).
    Both can be set; the index wins when both are configured.
    """

    name: str = "search_history"

    def __init__(
        self,
        *,
        store: SessionStore | None = None,
        index: SessionIndex | None = None,
        default_k: int = 5,
    ) -> None:
        """Bind a history search backend; at least one of store/index is required.

        Args:
            store: Session store for linear history scans.
            index: Optional FTS5-backed cross-session index (preferred).
            default_k: Result count used when the caller omits ``limit``.
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
        """Search past sessions for content matching *query*.

        Use this tool when the user references something they discussed
        in a previous session (e.g. "remember when we set up that
        webhook?"). The tool returns a list of relevant turn snippets
        with their session IDs, agent IDs, and timestamps so the agent
        can answer with provenance.

        Args:
            query: Free-text search.
            limit: Max results (default 5).
            agent_id: Optional filter by agent.
            session_id: Optional filter by session.

        Returns:
            Dict with:
                - ``query``: echo of the input query.
                - ``count``: number of hits returned.
                - ``hits``: list of dicts with ``session_id``,
                  ``turn_id``, ``agent_id``, ``prompt``, ``response``,
                  ``score``, ``timestamp``.
        """
        k = limit or self.default_k
        if self.index is not None:
            hits: list[SearchHit] = self.index.search(  # type: ignore[name-defined]
                query, k=k, agent_id=agent_id, session_id=session_id
            )
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
