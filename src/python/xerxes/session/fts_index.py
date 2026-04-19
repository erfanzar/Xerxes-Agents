# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""FTS5 full-text search index for session transcripts.

Provides fast full-text search across all session turns using SQLite FTS5.
Falls back to linear scan if FTS5 is unavailable in the SQLite build.

Usage::

    from xerxes.session.fts_index import SessionFTSIndex

    index = SessionFTSIndex("~/.xerxes/sessions/fts.db")
    index.index_session(session_record)
    results = index.search("deployment", k=10)
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .models import SessionRecord

logger = logging.getLogger(__name__)


class SessionFTSIndex:
    """FTS5-backed full-text search for session transcripts.

    Each turn's prompt + response is indexed as a single document.
    Sessions are re-indexed on every call to :meth:`index_session` —
    existing turns for that session are deleted first, then re-inserted.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fts_available = self._check_fts5()
        if self._fts_available:
            self._ensure_schema()

    def _check_fts5(self) -> bool:
        """Return True if the SQLite build supports FTS5."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_fts5_test'")
            cur.fetchall()
            conn.close()

            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE VIRTUAL TABLE fts_test USING fts5(content)")
            conn.close()
            return True
        except Exception as exc:
            logger.warning("FTS5 not available (%s). Session search will use linear scan.", exc)
            return False

    def _ensure_schema(self) -> None:
        """Create FTS5 virtual table if missing."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS session_fts USING fts5(
                    session_id,
                    turn_id,
                    agent_id,
                    content
                )
            """
            )
            conn.commit()

    def index_session(self, session: SessionRecord) -> None:
        """Index (or re-index) all turns from *session*.

        Deletes existing turns for this session_id, then inserts fresh rows.

        Args:
            session: The session record to index.
        """
        if not self._fts_available:
            return

        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "DELETE FROM session_fts WHERE session_id = ?",
                (session.session_id,),
            )

            for turn in session.turns:
                content = f"{turn.prompt or ''}\n{turn.response_content or ''}".strip()
                if not content:
                    continue
                conn.execute(
                    "INSERT INTO session_fts (session_id, turn_id, agent_id, content) VALUES (?, ?, ?, ?)",
                    (session.session_id, turn.turn_id, turn.agent_id or "", content),
                )
            conn.commit()

    def delete_session(self, session_id: str) -> None:
        """Remove all indexed turns for *session_id*.

        Args:
            session_id: The session to remove from the index.
        """
        if not self._fts_available:
            return
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "DELETE FROM session_fts WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search indexed session turns matching *query*.

        Args:
            query: Free-text search query.
            k: Maximum number of results.
            agent_id: Optional agent filter.
            session_id: Optional session filter.

        Returns:
            List of result dicts with keys ``session_id``, ``turn_id``,
            ``agent_id``, ``content``, ``rank``.
        """
        if not self._fts_available or not query.strip():
            return []

        sql = """
            SELECT session_id, turn_id, agent_id, content, rank
            FROM session_fts
            WHERE session_fts MATCH ?
        """
        params: list[Any] = [query]

        if agent_id is not None:
            sql += " AND agent_id = ?"
            params.append(agent_id)
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)

        sql += " ORDER BY rank LIMIT ?"
        params.append(k)

        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(sql, params)
            rows = cur.fetchall()

        return [
            {
                "session_id": row["session_id"],
                "turn_id": row["turn_id"],
                "agent_id": row["agent_id"],
                "content": row["content"],
                "rank": row["rank"],
            }
            for row in rows
        ]
