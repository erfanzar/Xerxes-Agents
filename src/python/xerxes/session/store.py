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


"""Session storage backends and high-level session manager.

Provides an abstract SessionStore protocol with two concrete implementations
(in-memory and file-based) and a SessionManager for session lifecycle management.
"""

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
    """Abstract base class for session storage backends.

    All concrete session stores must implement the four CRUD methods defined
    here: :meth:`save_session`, :meth:`load_session`, :meth:`list_sessions`,
    and :meth:`delete_session`. Implementations are expected to be
    thread-safe.
    """

    @abstractmethod
    def save_session(self, session: SessionRecord) -> None:
        """Persist a session record.

        Args:
            session: The session record to save.
        """

    @abstractmethod
    def load_session(self, session_id: str) -> SessionRecord | None:
        """Load a session record by ID.

        Args:
            session_id: The unique session identifier.

        Returns:
            The session record, or None if not found.
        """

    @abstractmethod
    def list_sessions(self, workspace_id: str | None = None) -> list[str]:
        """List session IDs, optionally filtered by workspace.

        Args:
            workspace_id: If provided, only return sessions in this workspace.

        Returns:
            List of session ID strings.
        """

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete a session record.

        Args:
            session_id: The unique session identifier.

        Returns:
            True if a session was deleted, False if not found.
        """

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> list:
        """Cross-session search across all turns.

        Default implementation does a linear scan over every session
        loaded from the store. For large stores, attach a
        :class:`~xerxes.session.index.SessionIndex` and override.

        Args:
            query: Free-text search.
            k: Maximum number of results.
            agent_id: Optional agent filter.
            session_id: Optional session filter.

        Returns:
            A list of :class:`~xerxes.session.index.SearchHit` items.
        """
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
    """Thread-safe in-memory session store backed by a dictionary.

    Sessions are kept in a plain ``dict`` protected by a :class:`threading.Lock`.
    Data does not survive process restarts; use :class:`FileSessionStore` for
    persistent storage.

    Attributes:
        _sessions: Internal dictionary mapping session IDs to records.
        _lock: Threading lock for safe concurrent access.

    Example:
        >>> store = InMemorySessionStore()
        >>> store.list_sessions()
        []
    """

    def __init__(self) -> None:
        """Initialise an empty in-memory session store with a threading lock."""
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    def save_session(self, session: SessionRecord) -> None:
        """Save or overwrite a session record in memory.

        If a session with the same ``session_id`` already exists, it is
        silently replaced.

        Args:
            session: The session record to store.
        """
        with self._lock:
            self._sessions[session.session_id] = session

    def load_session(self, session_id: str) -> SessionRecord | None:
        """Load a session record from memory by its identifier.

        Args:
            session_id: The unique session identifier to look up.

        Returns:
            The matching ``SessionRecord``, or ``None`` if no session with
            the given ID exists.
        """
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self, workspace_id: str | None = None) -> list[str]:
        """List session IDs stored in memory, optionally filtered by workspace.

        Args:
            workspace_id: When provided, only session IDs belonging to this
                workspace are returned. When ``None``, all session IDs are
                returned.

        Returns:
            A list of session ID strings.
        """
        with self._lock:
            if workspace_id is None:
                return list(self._sessions.keys())
            return [sid for sid, s in self._sessions.items() if s.workspace_id == workspace_id]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session record from memory.

        Args:
            session_id: The unique session identifier to delete.

        Returns:
            ``True`` if a session was found and deleted, ``False`` if no
            session with the given ID existed.
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False


class FileSessionStore(SessionStore):
    """File-backed session store using JSON files.

    Each session is persisted as an individual JSON file on disk. The
    directory layout is determined by whether the session has a workspace ID:

    Directory layout::

        {base_dir}/{session_id}.json                  -- workspace_id is None
        {base_dir}/{workspace_id}/{session_id}.json   -- workspace_id is set

    Thread-safe via an internal :class:`threading.Lock`.

    Attributes:
        _base_dir: Root directory for session JSON files.
        _lock: Threading lock for safe concurrent access.

    Example:
        >>> import tempfile
        >>> store = FileSessionStore(tempfile.mkdtemp())
        >>> store.list_sessions()
        []
    """

    def __init__(self, base_dir: str | Path) -> None:
        """Initialise the file session store.

        Creates the *base_dir* directory (and any parents) if it does not
        already exist.

        Args:
            base_dir: Root directory path where session JSON files will be
                stored. Accepts a string or :class:`~pathlib.Path`.
        """
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _session_path(self, session: SessionRecord) -> Path:
        """Resolve the file path for a given session record.

        Sessions with a ``workspace_id`` are nested under a workspace
        subdirectory; sessions without one are stored directly under
        *base_dir*.

        Args:
            session: The session record whose file path is needed.

        Returns:
            The :class:`~pathlib.Path` where the session JSON file should
            be written.
        """
        if session.workspace_id:
            directory = self._base_dir / session.workspace_id
        else:
            directory = self._base_dir
        return directory / f"{session.session_id}.json"

    def _find_session_path(self, session_id: str) -> Path | None:
        """Find the file path for a session ID by searching the directory tree.

        First checks for a flat file at ``{base_dir}/{session_id}.json``, then
        searches one level of workspace subdirectories.

        Args:
            session_id: The session identifier to locate on disk.

        Returns:
            The :class:`~pathlib.Path` to the JSON file if found, otherwise
            ``None``.
        """
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
        """Save a session record as a JSON file on disk.

        Creates the parent directory (including workspace subdirectories)
        if it does not already exist. Existing files are overwritten.

        Args:
            session: The session record to persist.
        """
        with self._lock:
            path = self._session_path(session)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")

    def load_session(self, session_id: str) -> SessionRecord | None:
        """Load a session record from a JSON file on disk.

        Searches the directory tree for a matching JSON file, reads it,
        and deserializes it into a ``SessionRecord``.

        Args:
            session_id: The unique session identifier to load.

        Returns:
            The deserialized ``SessionRecord``, or ``None`` if no matching
            file is found.
        """
        with self._lock:
            path = self._find_session_path(session_id)
            if path is None:
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            return SessionRecord.from_dict(data)

    def list_sessions(self, workspace_id: str | None = None) -> list[str]:
        """List session IDs by scanning JSON files on disk.

        When *workspace_id* is provided, only the corresponding workspace
        subdirectory is scanned. Otherwise, both the flat base directory
        and all workspace subdirectories are scanned.

        Args:
            workspace_id: When provided, restricts the scan to the
                ``{base_dir}/{workspace_id}/`` subdirectory only. When
                ``None``, all JSON files across all directories are included.

        Returns:
            A list of session ID strings (JSON file stems).
        """
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
        """Delete a session JSON file from disk.

        Searches the directory tree for the matching file and removes it.

        Args:
            session_id: The unique session identifier to delete.

        Returns:
            ``True`` if the file was found and deleted, ``False`` if no
            matching file existed.
        """
        with self._lock:
            path = self._find_session_path(session_id)
            if path is None:
                return False
            path.unlink()
            return True


class SessionManager:
    """High-level API for session lifecycle management.

    Wraps a :class:`SessionStore` to provide convenient methods for creating,
    recording turns, recording agent transitions, and ending sessions. Each
    mutation automatically updates the ``updated_at`` timestamp and persists
    the session back to the store.

    Attributes:
        _store: The underlying session store backend used for persistence.

    Example:
        >>> store = InMemorySessionStore()
        >>> manager = SessionManager(store)
        >>> session = manager.start_session(agent_id="default")
        >>> session.agent_id
        'default'
    """

    def __init__(self, store: SessionStore) -> None:
        """Initialise the session manager with a storage backend.

        Args:
            store: The :class:`SessionStore` implementation to delegate
                persistence operations to.
        """
        self._store = store

    @property
    def store(self) -> SessionStore:
        """The underlying session store."""
        return self._store

    def start_session(
        self,
        workspace_id: str | None = None,
        agent_id: str | None = None,
        *,
        session_id: str | None = None,
        metadata: dict[str, tp.Any] | None = None,
    ) -> SessionRecord:
        """Create and persist a new session.

        Args:
            workspace_id: Workspace to associate the session with.
            agent_id: Initial agent for the session.
            session_id: Explicit session ID; auto-generated if omitted.
            metadata: Optional metadata dict.

        Returns:
            The newly created SessionRecord.
        """
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
        """Append a turn record to an existing session.

        Args:
            session_id: The session to append to.
            turn: The turn record to add.

        Raises:
            ValueError: If the session does not exist.
        """
        session = self._store.load_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        session.turns.append(turn)
        session.updated_at = datetime.now(UTC).isoformat()
        self._store.save_session(session)

    def record_agent_transition(self, session_id: str, transition: AgentTransitionRecord) -> None:
        """Record an agent transition in a session.

        Args:
            session_id: The session to record the transition in.
            transition: The transition record.

        Raises:
            ValueError: If the session does not exist.
        """
        session = self._store.load_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        session.agent_transitions.append(transition)
        session.updated_at = datetime.now(UTC).isoformat()
        self._store.save_session(session)

    def end_session(self, session_id: str) -> None:
        """Mark a session as ended by updating its timestamp.

        Args:
            session_id: The session to end.

        Raises:
            ValueError: If the session does not exist.
        """
        session = self._store.load_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        session.updated_at = datetime.now(UTC).isoformat()
        session.metadata["ended"] = True
        self._store.save_session(session)
        logger.debug("Ended session %s", session_id)

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Retrieve a session record.

        Args:
            session_id: The session to retrieve.

        Returns:
            The session record, or None if not found.
        """
        return self._store.load_session(session_id)

    def list_sessions(self, workspace_id: str | None = None) -> list[str]:
        """List session IDs.

        Args:
            workspace_id: If provided, filter by workspace.

        Returns:
            List of session ID strings.
        """
        return self._store.list_sessions(workspace_id)
