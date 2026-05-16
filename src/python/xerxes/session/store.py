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
"""Session record persistence backends.

Defines the abstract :class:`SessionStore` plus two concrete implementations:

* :class:`InMemorySessionStore` — pure-Python dict guarded by a lock. Used in
  tests and short-lived processes; data evaporates on shutdown.
* :class:`FileSessionStore` — one JSON file per session on disk, written
  atomically via tempfile + ``os.replace``. Records are migrated forward on
  load and rewritten in-place if their schema version was bumped.

:class:`SessionManager` is the thin lifecycle wrapper agents use to start a
session, append turns, and record agent transitions without hand-rolling the
``updated_at`` bookkeeping.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import typing as tp
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

from .migrations import migrate_record
from .models import CURRENT_SCHEMA_VERSION, AgentTransitionRecord, SessionRecord, TurnRecord

logger = logging.getLogger(__name__)


class SessionStore(ABC):
    """Abstract persistence interface for :class:`SessionRecord` objects.

    Concrete subclasses must provide save/load/list/delete and are expected to
    be safe to call from multiple threads. The default :meth:`search`
    implementation is a linear-scan fallback for stores that have no native
    index; richer backends (see :mod:`xerxes.session.fts_index`) should
    override it.
    """

    @abstractmethod
    def save_session(self, session: SessionRecord) -> None:
        """Persist a session record, overwriting any previous version."""
        ...

    @abstractmethod
    def load_session(self, session_id: str) -> SessionRecord | None:
        """Return the stored record for ``session_id`` or ``None`` if absent."""
        ...

    @abstractmethod
    def list_sessions(self, workspace_id: str | None = None) -> list[str]:
        """Return all known session ids, optionally scoped to one workspace."""
        ...

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete the session and return whether it existed."""
        ...

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> list:
        """Substring-match ``query`` against turn prompts and responses.

        The default backend has no index — it linearly walks every session and
        every turn until it has collected ``k`` hits. Subclasses with FTS or
        vector indexes should override this for non-trivial corpora.

        Args:
            query: Case-insensitive substring to match.
            k: Maximum number of hits to return.
            agent_id: Restrict matches to turns produced by this agent.
            session_id: Restrict matches to a single session.

        Returns:
            List of :class:`SearchHit` ordered as they are discovered (the
            naive scan does not rank).
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
    """Volatile :class:`SessionStore` backed by an in-process dict.

    All operations are guarded by a single :class:`threading.Lock`, so the
    store is safe to share between threads. Records vanish when the process
    exits — use :class:`FileSessionStore` for anything you need to outlive a
    crash.
    """

    def __init__(self) -> None:
        """Create an empty in-memory store."""

        self._sessions: dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    def save_session(self, session: SessionRecord) -> None:
        """Insert or replace ``session`` keyed by its id."""

        with self._lock:
            self._sessions[session.session_id] = session

    def load_session(self, session_id: str) -> SessionRecord | None:
        """Return the in-memory record or ``None`` if no such id exists."""

        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self, workspace_id: str | None = None) -> list[str]:
        """Return all session ids, filtered by ``workspace_id`` when given."""

        with self._lock:
            if workspace_id is None:
                return list(self._sessions.keys())
            return [sid for sid, s in self._sessions.items() if s.workspace_id == workspace_id]

    def delete_session(self, session_id: str) -> bool:
        """Drop ``session_id``; return ``True`` if it had been present."""

        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False


class FileSessionStore(SessionStore):
    """Disk-backed :class:`SessionStore` storing one JSON file per session.

    Layout: ``<base_dir>/<workspace_id>/<session_id>.json``, falling back to
    ``<base_dir>/<session_id>.json`` for sessions without a workspace. Writes
    are atomic (tempfile + ``os.replace``) so a crash mid-save cannot leave a
    half-written file. A single process-wide lock serialises operations; this
    is enough for the daemon, but external processes touching the same files
    are not synchronised.
    """

    def __init__(self, base_dir: str | Path) -> None:
        """Create the store rooted at ``base_dir`` (created if missing)."""

        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _session_path(self, session: SessionRecord) -> Path:
        """Return the on-disk path for ``session`` (workspace-aware)."""

        if session.workspace_id:
            directory = self._base_dir / session.workspace_id
        else:
            directory = self._base_dir
        return directory / f"{session.session_id}.json"

    def _find_session_path(self, session_id: str) -> Path | None:
        """Locate an existing session file under either layout."""

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
        """Atomically write ``session`` to its workspace-scoped JSON file.

        Serialises to a tempfile in the same directory then ``os.replace``s
        onto the final path so a crash mid-write cannot leave a partial file.
        The tempfile is cleaned up if writing raises.
        """

        with self._lock:
            path = self._session_path(session)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(session.to_dict(), indent=2)
            fd, tmp_path = tempfile.mkstemp(prefix=f".{session.session_id}.", suffix=".json", dir=str(path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(payload)
                os.replace(tmp_path, path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except FileNotFoundError:
                    pass
                raise

    def load_session(self, session_id: str) -> SessionRecord | None:
        """Load and forward-migrate the session file for ``session_id``.

        Forward-only migrations are applied until the on-disk
        ``schema_version`` reaches :data:`CURRENT_SCHEMA_VERSION`. If any
        migration ran, the upgraded payload is written back atomically so
        subsequent loads are zero-cost. Returns ``None`` when no file exists.
        """

        with self._lock:
            path = self._find_session_path(session_id)
            if path is None:
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            stored_version = int(data.get("schema_version", 1))
            if stored_version < CURRENT_SCHEMA_VERSION:
                data = migrate_record(data, CURRENT_SCHEMA_VERSION)
                # Persist the upgraded form. Use the same atomic-write path.
                payload = json.dumps(data, indent=2)
                fd, tmp_path = tempfile.mkstemp(prefix=f".{session_id}.", suffix=".json", dir=str(path.parent))
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as fh:
                        fh.write(payload)
                    os.replace(tmp_path, path)
                except Exception:
                    try:
                        os.unlink(tmp_path)
                    except FileNotFoundError:
                        pass
                    raise
            return SessionRecord.from_dict(data)

    def list_sessions(self, workspace_id: str | None = None) -> list[str]:
        """Walk the base dir and return every session id found.

        When ``workspace_id`` is given, only that sub-directory is scanned;
        otherwise both the flat root files and every nested workspace folder
        are included.
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
        """Unlink the session's JSON file; return ``True`` if one existed."""

        with self._lock:
            path = self._find_session_path(session_id)
            if path is None:
                return False
            path.unlink()
            return True


class SessionManager:
    """Lifecycle wrapper around a :class:`SessionStore`.

    Owns the bookkeeping the daemon would otherwise duplicate per call site:
    generating session ids, stamping ``created_at`` / ``updated_at``, and
    persisting every mutation. The underlying store is exposed via
    :attr:`store` for read paths that need the raw API.
    """

    def __init__(self, store: SessionStore) -> None:
        """Wrap ``store`` and route every mutation through it."""

        self._store = store

    @property
    def store(self) -> SessionStore:
        """The backing :class:`SessionStore`."""

        return self._store

    def start_session(
        self,
        workspace_id: str | None = None,
        agent_id: str | None = None,
        *,
        session_id: str | None = None,
        metadata: dict[str, tp.Any] | None = None,
    ) -> SessionRecord:
        """Create, persist, and return a fresh :class:`SessionRecord`.

        Args:
            workspace_id: Workspace the session belongs to, used for layout
                on disk and filtering on listing.
            agent_id: Initial agent that owns the session.
            session_id: Override the generated UUID hex (mostly for tests).
            metadata: Arbitrary JSON-serialisable metadata seeded into the
                record.
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
        """Append ``turn`` to the session and persist.

        Raises:
            ValueError: If ``session_id`` is unknown.
        """

        session = self._store.load_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        session.turns.append(turn)
        session.updated_at = datetime.now(UTC).isoformat()
        self._store.save_session(session)

    def record_agent_transition(self, session_id: str, transition: AgentTransitionRecord) -> None:
        """Append an agent-handoff record and persist.

        Raises:
            ValueError: If ``session_id`` is unknown.
        """

        session = self._store.load_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        session.agent_transitions.append(transition)
        session.updated_at = datetime.now(UTC).isoformat()
        self._store.save_session(session)

    def end_session(self, session_id: str) -> None:
        """Mark the session as ended in metadata and persist.

        The record is not deleted — ``metadata['ended']`` simply flips to
        ``True`` and ``updated_at`` advances.
        """

        session = self._store.load_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        session.updated_at = datetime.now(UTC).isoformat()
        session.metadata["ended"] = True
        self._store.save_session(session)
        logger.debug("Ended session %s", session_id)

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Return the stored record or ``None`` if absent."""

        return self._store.load_session(session_id)

    def list_sessions(self, workspace_id: str | None = None) -> list[str]:
        """Delegate to the store's ``list_sessions``."""

        return self._store.list_sessions(workspace_id)
