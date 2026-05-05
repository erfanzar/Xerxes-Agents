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
"""Store module for Xerxes.

Exports:
    - logger
    - SessionStore
    - InMemorySessionStore
    - FileSessionStore
    - SessionManager"""

from __future__ import annotations

import json
import logging
import threading
import typing as tp
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

from .models import AgentTransitionRecord, SessionRecord, TurnRecord

logger = logging.getLogger(__name__)


class SessionStore(ABC):
    """Session store.

    Inherits from: ABC
    """

    @abstractmethod
    def save_session(self, session: SessionRecord) -> None:
        """Save session.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session (SessionRecord): IN: session. OUT: Consumed during execution."""
        ...

    @abstractmethod
    def load_session(self, session_id: str) -> SessionRecord | None:
        """Load session.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session_id (str): IN: session id. OUT: Consumed during execution.
        Returns:
            SessionRecord | None: OUT: Result of the operation."""
        ...

    @abstractmethod
    def list_sessions(self, workspace_id: str | None = None) -> list[str]:
        """List sessions.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            workspace_id (str | None, optional): IN: workspace id. Defaults to None. OUT: Consumed during execution.
        Returns:
            list[str]: OUT: Result of the operation."""
        ...

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete session.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session_id (str): IN: session id. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""
        ...

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> list:
        """Search.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            query (str): IN: query. OUT: Consumed during execution.
            k (int, optional): IN: k. Defaults to 10. OUT: Consumed during execution.
            agent_id (str | None, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
            session_id (str | None, optional): IN: session id. Defaults to None. OUT: Consumed during execution.
        Returns:
            list: OUT: Result of the operation."""

        from .index import SearchHit

        if not query.strip():
            return []
        ql = query.lower()
        hits: list[SearchHit] = []
        ids = self.list_sessions()
        for sid in ids:
            if session_id is not None and sid != session_id:
                continue
            sess = self.load_session(sid)
            if sess is None:
                continue
            for turn in sess.turns:
                if agent_id is not None and turn.agent_id != agent_id:
                    continue
                blob = f"{turn.prompt or ''}\n{turn.response_content or ''}".lower()
                if ql in blob:
                    hits.append(
                        SearchHit(
                            session_id=sid,
                            turn_id=turn.turn_id,
                            agent_id=turn.agent_id,
                            prompt=(turn.prompt or "")[:500],
                            response=(turn.response_content or "")[:1000],
                            score=1.0,
                            bm25_score=1.0,
                            timestamp=turn.started_at or "",
                        )
                    )
                    if len(hits) >= k:
                        return hits
        return hits


class InMemorySessionStore(SessionStore):
    """In memory session store.

    Inherits from: SessionStore
    """

    def __init__(self) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        self._sessions: dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    def save_session(self, session: SessionRecord) -> None:
        """Save session.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session (SessionRecord): IN: session. OUT: Consumed during execution."""

        with self._lock:
            self._sessions[session.session_id] = session

    def load_session(self, session_id: str) -> SessionRecord | None:
        """Load session.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session_id (str): IN: session id. OUT: Consumed during execution.
        Returns:
            SessionRecord | None: OUT: Result of the operation."""

        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self, workspace_id: str | None = None) -> list[str]:
        """List sessions.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            workspace_id (str | None, optional): IN: workspace id. Defaults to None. OUT: Consumed during execution.
        Returns:
            list[str]: OUT: Result of the operation."""

        with self._lock:
            if workspace_id is None:
                return list(self._sessions.keys())
            return [sid for sid, s in self._sessions.items() if s.workspace_id == workspace_id]

    def delete_session(self, session_id: str) -> bool:
        """Delete session.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session_id (str): IN: session id. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False


class FileSessionStore(SessionStore):
    """File session store.

    Inherits from: SessionStore
    """

    def __init__(self, base_dir: str | Path) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            base_dir (str | Path): IN: base dir. OUT: Consumed during execution."""

        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _session_path(self, session: SessionRecord) -> Path:
        """Internal helper to session path.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session (SessionRecord): IN: session. OUT: Consumed during execution.
        Returns:
            Path: OUT: Result of the operation."""

        if session.workspace_id:
            directory = self._base_dir / session.workspace_id
        else:
            directory = self._base_dir
        return directory / f"{session.session_id}.json"

    def _find_session_path(self, session_id: str) -> Path | None:
        """Internal helper to find session path.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session_id (str): IN: session id. OUT: Consumed during execution.
        Returns:
            Path | None: OUT: Result of the operation."""

        flat = self._base_dir / f"{session_id}.json"
        if flat.exists():
            return flat
        for child in self._base_dir.iterdir():
            if child.is_dir():
                candidate = child / f"{session_id}.json"
                if candidate.exists():
                    return candidate
        return None

    def save_session(self, session: SessionRecord) -> None:
        """Save session.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session (SessionRecord): IN: session. OUT: Consumed during execution."""

        with self._lock:
            path = self._session_path(session)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")

    def load_session(self, session_id: str) -> SessionRecord | None:
        """Load session.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session_id (str): IN: session id. OUT: Consumed during execution.
        Returns:
            SessionRecord | None: OUT: Result of the operation."""

        with self._lock:
            path = self._find_session_path(session_id)
            if path is None:
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            return SessionRecord.from_dict(data)

    def list_sessions(self, workspace_id: str | None = None) -> list[str]:
        """List sessions.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            workspace_id (str | None, optional): IN: workspace id. Defaults to None. OUT: Consumed during execution.
        Returns:
            list[str]: OUT: Result of the operation."""

        with self._lock:
            results: list[str] = []
            if workspace_id is not None:
                search_dir = self._base_dir / workspace_id
                if not search_dir.is_dir():
                    return []
                for f in search_dir.glob("*.json"):
                    results.append(f.stem)
            else:
                for f in self._base_dir.glob("*.json"):
                    results.append(f.stem)
                for child in self._base_dir.iterdir():
                    if child.is_dir():
                        for f in child.glob("*.json"):
                            results.append(f.stem)
            return results

    def delete_session(self, session_id: str) -> bool:
        """Delete session.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session_id (str): IN: session id. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        with self._lock:
            path = self._find_session_path(session_id)
            if path is None:
                return False
            path.unlink()
            return True


class SessionManager:
    """Session manager."""

    def __init__(self, store: SessionStore) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            store (SessionStore): IN: store. OUT: Consumed during execution."""

        self._store = store

    @property
    def store(self) -> SessionStore:
        """Return Store.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            SessionStore: OUT: Result of the operation."""

        return self._store

    def start_session(
        self,
        workspace_id: str | None = None,
        agent_id: str | None = None,
        *,
        session_id: str | None = None,
        metadata: dict[str, tp.Any] | None = None,
    ) -> SessionRecord:
        """Start session.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            workspace_id (str | None, optional): IN: workspace id. Defaults to None. OUT: Consumed during execution.
            agent_id (str | None, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
            session_id (str | None, optional): IN: session id. Defaults to None. OUT: Consumed during execution.
            metadata (dict[str, tp.Any] | None, optional): IN: metadata. Defaults to None. OUT: Consumed during execution.
        Returns:
            SessionRecord: OUT: Result of the operation."""

        now = datetime.now(UTC).isoformat()
        session = SessionRecord(
            session_id=session_id or uuid.uuid4().hex,
            workspace_id=workspace_id,
            created_at=now,
            updated_at=now,
            agent_id=agent_id,
            metadata=metadata or {},
        )
        self._store.save_session(session)
        logger.debug("Started session %s", session.session_id)
        return session

    def record_turn(self, session_id: str, turn: TurnRecord) -> None:
        """Record turn.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session_id (str): IN: session id. OUT: Consumed during execution.
            turn (TurnRecord): IN: turn. OUT: Consumed during execution."""

        session = self._store.load_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        session.turns.append(turn)
        session.updated_at = datetime.now(UTC).isoformat()
        self._store.save_session(session)

    def record_agent_transition(self, session_id: str, transition: AgentTransitionRecord) -> None:
        """Record agent transition.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session_id (str): IN: session id. OUT: Consumed during execution.
            transition (AgentTransitionRecord): IN: transition. OUT: Consumed during execution."""

        session = self._store.load_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        session.agent_transitions.append(transition)
        session.updated_at = datetime.now(UTC).isoformat()
        self._store.save_session(session)

    def end_session(self, session_id: str) -> None:
        """End session.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session_id (str): IN: session id. OUT: Consumed during execution."""

        session = self._store.load_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        session.updated_at = datetime.now(UTC).isoformat()
        session.metadata["ended"] = True
        self._store.save_session(session)
        logger.debug("Ended session %s", session_id)

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Retrieve the session.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session_id (str): IN: session id. OUT: Consumed during execution.
        Returns:
            SessionRecord | None: OUT: Result of the operation."""

        return self._store.load_session(session_id)

    def list_sessions(self, workspace_id: str | None = None) -> list[str]:
        """List sessions.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            workspace_id (str | None, optional): IN: workspace id. Defaults to None. OUT: Consumed during execution.
        Returns:
            list[str]: OUT: Result of the operation."""

        return self._store.list_sessions(workspace_id)
