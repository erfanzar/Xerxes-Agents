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
"""Fts index module for Xerxes.

Exports:
    - logger
    - SessionFTSIndex"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .models import SessionRecord

logger = logging.getLogger(__name__)


class SessionFTSIndex:
    """Session ftsindex."""

    def __init__(self, db_path: str | Path) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            db_path (str | Path): IN: db path. OUT: Consumed during execution."""
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fts_available = self._check_fts5()
        if self._fts_available:
            self._ensure_schema()

    def _check_fts5(self) -> bool:
        """Internal helper to check fts5.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            bool: OUT: Result of the operation."""

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
        """Internal helper to ensure schema.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

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
        """Index session.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session (SessionRecord): IN: session. OUT: Consumed during execution."""

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
        """Delete session.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session_id (str): IN: session id. OUT: Consumed during execution."""

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
        """Search.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            query (str): IN: query. OUT: Consumed during execution.
            k (int, optional): IN: k. Defaults to 10. OUT: Consumed during execution.
            agent_id (str | None, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
            session_id (str | None, optional): IN: session id. Defaults to None. OUT: Consumed during execution.
        Returns:
            list[dict[str, Any]]: OUT: Result of the operation."""

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
