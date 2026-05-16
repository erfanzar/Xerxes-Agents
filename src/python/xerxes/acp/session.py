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
"""ACP session model + persistence.

ACP sessions are thin wrappers over Xerxes ``SessionRecord``s. The
ACP layer adds a per-session ``cwd`` (the directory the client is
working in) and a ``model_override`` (client-selectable model)."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..session.models import SessionRecord


@dataclass
class AcpSession:
    """Live ACP session record.

    Attributes:
        session_id: Xerxes ``SessionRecord.session_id``.
        cwd: working directory the IDE client is in.
        model_override: optional model id the client picked.
        title: short human label.
        cancelled: ``True`` after a cancel arrives.
        metadata: free-form client annotations."""

    session_id: str
    cwd: str
    model_override: str | None = None
    title: str = ""
    cancelled: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class AcpSessionStore:
    """Thread-safe registry of live ACP sessions."""

    def __init__(self) -> None:
        """Start with an empty registry and a single mutex."""
        self._sessions: dict[str, AcpSession] = {}
        self._lock = threading.Lock()

    def create(self, cwd: str, *, model: str | None = None, title: str = "") -> AcpSession:
        """Create a fresh session with a uuid4 hex id and register it."""
        sid = uuid.uuid4().hex
        sess = AcpSession(session_id=sid, cwd=cwd, model_override=model, title=title)
        with self._lock:
            self._sessions[sid] = sess
        return sess

    def attach_existing(self, record: SessionRecord, *, cwd: str) -> AcpSession:
        """Wrap an existing Xerxes ``SessionRecord`` for ACP exposure."""
        sess = AcpSession(session_id=record.session_id, cwd=cwd)
        with self._lock:
            self._sessions[record.session_id] = sess
        return sess

    def get(self, session_id: str) -> AcpSession | None:
        """Return the session or None if unknown."""
        with self._lock:
            return self._sessions.get(session_id)

    def list(self) -> list[AcpSession]:
        """Snapshot of every live session."""
        with self._lock:
            return list(self._sessions.values())

    def cancel(self, session_id: str) -> bool:
        """Flip ``cancelled=True``; return False if the session is unknown."""
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                return False
            sess.cancelled = True
            return True

    def drop(self, session_id: str) -> bool:
        """Remove the session; return True iff it existed."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def set_model(self, session_id: str, model: str | None) -> bool:
        """Update the per-session model override; False if unknown."""
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                return False
            sess.model_override = model
            return True


__all__ = ["AcpSession", "AcpSessionStore"]
