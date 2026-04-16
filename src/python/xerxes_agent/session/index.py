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
"""Cross-session search index — SQLite FTS5 + optional vector recall.

Provides :class:`SessionIndex` which persists one row per ``TurnRecord``
into an FTS5-backed SQLite database. Hooks lazily fall back to a plain
LIKE query when SQLite was built without FTS5. An optional
:class:`Embedder` is consulted in addition to FTS to produce a hybrid
ranking similar to :class:`HybridRetriever`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import typing as tp
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..memory.embedders import Embedder, cosine_similarity
from .models import SessionRecord, TurnRecord

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    """One matched turn returned by :meth:`SessionIndex.search`.

    Attributes:
        session_id: Originating session.
        turn_id: Originating turn.
        agent_id: Agent that handled the turn.
        prompt: User prompt of the turn.
        response: Agent response (truncated to 1k chars).
        score: Composite score in ``[0, 1]``.
        bm25_score: FTS rank component.
        semantic_score: Cosine similarity component (0 when no embedder).
        timestamp: ``started_at`` from the turn.
    """

    session_id: str
    turn_id: str
    agent_id: str | None
    prompt: str
    response: str
    score: float
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    timestamp: str = ""
    metadata: dict[str, tp.Any] = field(default_factory=dict)


class SessionIndex:
    """Cross-session searchable index.

    The index lives in a single SQLite file. Schema:

    .. code-block:: sql

        CREATE TABLE turns (
            session_id TEXT,
            turn_id TEXT PRIMARY KEY,
            agent_id TEXT,
            prompt TEXT,
            response TEXT,
            started_at TEXT,
            embedding TEXT
        )

        CREATE VIRTUAL TABLE turns_fts USING fts5(
            prompt, response, content='turns', content_rowid='rowid'
        )

    When FTS5 is unavailable, the index degrades to LIKE matches.

    Example:
        >>> idx = SessionIndex(":memory:", embedder=HashEmbedder())
        >>> idx.index_session(record)
        >>> hits = idx.search("authentication bug", k=10)
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        embedder: Embedder | None = None,
    ) -> None:
        """Initialise the index, opening the SQLite file.

        Args:
            db_path: SQLite database path or ``:memory:``.
            embedder: Optional embedder for hybrid scoring. When ``None``,
                purely lexical search is used.
        """
        if isinstance(db_path, Path):
            db_path = str(db_path)
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.embedder = embedder
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._has_fts = self._init_schema()

    def _init_schema(self) -> bool:
        """Create the ``turns`` table and (if available) the FTS5 mirror.

        Issues ``CREATE TABLE IF NOT EXISTS`` for the main ``turns`` table
        and attempts ``CREATE VIRTUAL TABLE ... USING fts5``. The FTS
        creation is wrapped in a try/except so SQLite builds without FTS5
        degrade gracefully.

        Returns:
            ``True`` if FTS5 is usable; ``False`` when the index must
            fall back to LIKE-based search.
        """
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
                session_id TEXT,
                turn_id TEXT PRIMARY KEY,
                agent_id TEXT,
                prompt TEXT,
                response TEXT,
                started_at TEXT,
                metadata TEXT,
                embedding TEXT
            )
            """
        )
        has_fts = True
        try:
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
                    prompt, response, turn_id UNINDEXED, tokenize = 'porter unicode61'
                )
                """
            )
        except sqlite3.OperationalError:
            has_fts = False
            logger.info("SQLite FTS5 unavailable; SessionIndex will use LIKE search")
        self._conn.commit()
        return has_fts

    def close(self) -> None:
        """Close the underlying SQLite connection, suppressing any error."""
        try:
            self._conn.close()
        except Exception:
            pass

    def index_session(self, session: SessionRecord) -> int:
        """Index every turn in *session*; returns number of turns indexed."""
        n = 0
        for turn in session.turns:
            self.index_turn(session.session_id, turn)
            n += 1
        return n

    def index_turn(self, session_id: str, turn: TurnRecord) -> None:
        """Insert (or replace) a single turn row.

        Computes an embedding when an embedder is configured. Persists
        the JSON-encoded vector for later hybrid scoring.
        """
        prompt = turn.prompt or ""
        response = turn.response_content or ""
        emb_json = ""
        if self.embedder is not None and (prompt or response):
            try:
                vec = self.embedder.embed(f"{prompt}\n{response}")
                emb_json = json.dumps(vec)
            except Exception:
                logger.debug("Embedder failed for turn %s", turn.turn_id, exc_info=True)
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO turns
                (session_id, turn_id, agent_id, prompt, response, started_at, metadata, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                turn.turn_id,
                turn.agent_id,
                prompt,
                response,
                turn.started_at or datetime.now().isoformat(),
                json.dumps(turn.metadata or {}),
                emb_json,
            ),
        )
        if self._has_fts:
            cur.execute("DELETE FROM turns_fts WHERE turn_id = ?", (turn.turn_id,))
            cur.execute(
                "INSERT INTO turns_fts (prompt, response, turn_id) VALUES (?, ?, ?)",
                (prompt, response, turn.turn_id),
            )
        self._conn.commit()

    def remove_session(self, session_id: str) -> int:
        """Delete every indexed turn belonging to *session_id*.

        Removes rows from the ``turns`` table and, when FTS5 is enabled,
        matches them out of the ``turns_fts`` mirror as well.

        Args:
            session_id: Identifier of the session whose turns should be
                evicted from the index.

        Returns:
            The number of turn rows removed.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT turn_id FROM turns WHERE session_id = ?", (session_id,))
        ids = [r["turn_id"] for r in cur.fetchall()]
        if not ids:
            return 0
        cur.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
        if self._has_fts:
            placeholders = ",".join("?" * len(ids))
            cur.execute(f"DELETE FROM turns_fts WHERE turn_id IN ({placeholders})", ids)
        self._conn.commit()
        return len(ids)

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        agent_id: str | None = None,
        session_id: str | None = None,
        weights: tuple[float, float] = (0.6, 0.4),
    ) -> list[SearchHit]:
        """Search across all indexed turns.

        Combines (when both available) FTS5 BM25 ranking and cosine
        similarity over the embedding vector.

        Args:
            query: Free-text search.
            k: Maximum number of results.
            agent_id: Optional filter by agent.
            session_id: Optional filter by session.
            weights: ``(bm25_weight, semantic_weight)`` — both
                normalised to ``[0, 1]`` before combining.

        Returns:
            List of :class:`SearchHit` sorted by descending score.
        """
        if not query.strip():
            return []
        bm25_w, sem_w = weights
        norm = bm25_w + sem_w
        if norm <= 0:
            bm25_w, sem_w = 0.5, 0.5
        else:
            bm25_w, sem_w = bm25_w / norm, sem_w / norm
        rows = self._fetch_candidates(query, k=k * 4, agent_id=agent_id, session_id=session_id)
        if not rows:
            return []
        max_bm25 = max((r["bm25"] for r in rows), default=0.0) or 1.0
        qvec: list[float] | None = None
        if self.embedder is not None and sem_w > 0:
            try:
                qvec = self.embedder.embed(query)
            except Exception:
                qvec = None
        results: list[SearchHit] = []
        for r in rows:
            bm = r["bm25"] / max_bm25 if max_bm25 else 0.0
            sem = 0.0
            if qvec is not None and r["embedding"]:
                try:
                    vec = json.loads(r["embedding"])
                    sem = max(0.0, cosine_similarity(qvec, vec))
                except Exception:
                    sem = 0.0
            score = bm25_w * bm + sem_w * sem
            results.append(
                SearchHit(
                    session_id=r["session_id"],
                    turn_id=r["turn_id"],
                    agent_id=r["agent_id"],
                    prompt=(r["prompt"] or "")[:500],
                    response=(r["response"] or "")[:1000],
                    bm25_score=bm,
                    semantic_score=sem,
                    score=score,
                    timestamp=r["started_at"] or "",
                    metadata=json.loads(r["metadata"] or "{}") if r["metadata"] else {},
                )
            )
        results.sort(key=lambda h: h.score, reverse=True)
        return results[:k]

    def _fetch_candidates(
        self,
        query: str,
        *,
        k: int,
        agent_id: str | None,
        session_id: str | None,
    ) -> list[dict[str, tp.Any]]:
        """Run the first-pass lexical retrieval for :meth:`search`.

        Attempts an FTS5 ``MATCH`` query joined back to the ``turns``
        table, ordered by BM25 rank. If FTS5 is unavailable, or if the
        FTS query fails, or if FTS returns nothing, falls back to a
        ``LIKE '%query%'`` scan ordered by ``started_at DESC``.

        Args:
            query: Free-text query string.
            k: Maximum number of candidate rows to retrieve.
            agent_id: Optional ``agent_id`` filter.
            session_id: Optional ``session_id`` filter.

        Returns:
            Candidate row dicts containing identifiers, text, embeddings,
            and a ``bm25`` score (``1.0`` for the LIKE fallback).
        """
        cur = self._conn.cursor()
        rows: list[dict[str, tp.Any]] = []
        if self._has_fts:
            extra = ""
            params: list[tp.Any] = [query]
            if agent_id is not None:
                extra += " AND t.agent_id = ?"
                params.append(agent_id)
            if session_id is not None:
                extra += " AND t.session_id = ?"
                params.append(session_id)
            params.append(k)
            sql = (
                "SELECT t.session_id, t.turn_id, t.agent_id, t.prompt, t.response, "
                "t.started_at, t.metadata, t.embedding, bm25(turns_fts) AS rank "
                "FROM turns_fts JOIN turns t ON t.turn_id = turns_fts.turn_id "
                f"WHERE turns_fts MATCH ?{extra} "
                "ORDER BY rank LIMIT ?"
            )
            try:
                cur.execute(sql, params)
                for row in cur.fetchall():
                    bm = -float(row["rank"]) if row["rank"] is not None else 0.0
                    rows.append(
                        {
                            "session_id": row["session_id"],
                            "turn_id": row["turn_id"],
                            "agent_id": row["agent_id"],
                            "prompt": row["prompt"],
                            "response": row["response"],
                            "started_at": row["started_at"],
                            "metadata": row["metadata"],
                            "embedding": row["embedding"],
                            "bm25": max(bm, 0.0),
                        }
                    )
            except sqlite3.OperationalError:
                rows = []
        if not rows:
            like_q = f"%{query}%"
            params = [like_q, like_q]
            extra = ""
            if agent_id is not None:
                extra += " AND agent_id = ?"
                params.append(agent_id)
            if session_id is not None:
                extra += " AND session_id = ?"
                params.append(session_id)
            cur.execute(
                f"""
                SELECT session_id, turn_id, agent_id, prompt, response, started_at,
                       metadata, embedding
                FROM turns
                WHERE (prompt LIKE ? OR response LIKE ?){extra}
                ORDER BY started_at DESC
                LIMIT ?
                """,
                [*params, k],
            )
            for row in cur.fetchall():
                rows.append(
                    {
                        "session_id": row["session_id"],
                        "turn_id": row["turn_id"],
                        "agent_id": row["agent_id"],
                        "prompt": row["prompt"],
                        "response": row["response"],
                        "started_at": row["started_at"],
                        "metadata": row["metadata"],
                        "embedding": row["embedding"],
                        "bm25": 1.0,
                    }
                )
        return rows


__all__ = ["SearchHit", "SessionIndex"]
