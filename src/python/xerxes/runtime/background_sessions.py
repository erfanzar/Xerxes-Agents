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
"""Detached, fire-and-forget background agent sessions.

The TUI's ``/background <prompt>`` slash command lands here. :class:`BackgroundSessionManager`
spawns a worker thread that runs the prompt through an injected :data:`RunFn`
(in production this wraps the daemon's ``query`` RPC), records the final
output on the :class:`BackgroundSession`, and lets the user poll status via
:meth:`BackgroundSessionManager.list_sessions` and :meth:`BackgroundSessionManager.get`.
The runner is injected so tests don't need a live daemon.
"""

from __future__ import annotations

import enum
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class BackgroundStatus(enum.Enum):
    """Lifecycle states of a :class:`BackgroundSession`.

    Attributes:
        PENDING: Submitted but waiting on a slot under the concurrency cap.
        RUNNING: A worker thread is executing the runner.
        SUCCEEDED: Runner returned normally; ``result`` holds its output.
        FAILED: Runner raised; ``error`` carries the exception text.
        CANCELLED: User-requested cancellation observed; the thread is
            allowed to finish its own work cooperatively.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackgroundSession:
    """Record describing one queued, running, or finished background session.

    Attributes:
        id: Stable 12-char session identifier.
        prompt: Text submitted for execution.
        status: Current :class:`BackgroundStatus`.
        result: Final output (populated when ``status`` is ``SUCCEEDED``).
        error: Exception text (populated when ``status`` is ``FAILED``).
        started_at: Epoch seconds when execution began (``0.0`` until then).
        finished_at: Epoch seconds when the session reached a terminal state.
        metadata: Free-form key/value bag for caller use.
    """

    id: str
    prompt: str
    status: BackgroundStatus = BackgroundStatus.PENDING
    result: str = ""
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


RunFn = Callable[[BackgroundSession], str]
"""Callable contract for the injected runner: ``(session) -> result_text``."""


class BackgroundSessionManager:
    """Spawn and track threads running background agent sessions."""

    def __init__(self, runner: RunFn, *, max_concurrent: int = 4) -> None:
        """Create a manager bound to ``runner`` with a concurrency cap.

        Args:
            runner: Callable invoked on a worker thread for each session.
                Its return value populates ``session.result``; exceptions are
                captured into ``session.error``.
            max_concurrent: Maximum number of sessions running simultaneously
                (clamped to ``>= 1``). Additional submissions stay ``PENDING``
                until a slot frees up.
        """
        self._runner = runner
        self._max = max(1, int(max_concurrent))
        self._sessions: dict[str, BackgroundSession] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def submit(self, prompt: str, *, session_id: str | None = None) -> BackgroundSession:
        """Queue a new background session and start it if a slot is free.

        Args:
            prompt: Prompt text the runner will execute.
            session_id: Optional pre-chosen id; a random hex slice is used
                otherwise.

        Returns:
            The freshly created :class:`BackgroundSession`. It may already be
            ``RUNNING`` when there is spare concurrency, or ``PENDING`` if
            the cap is full.

        Raises:
            RuntimeError: :meth:`shutdown` has already been called.
        """
        if self._stop.is_set():
            raise RuntimeError("manager is shutting down")
        sid = session_id or uuid.uuid4().hex[:12]
        sess = BackgroundSession(id=sid, prompt=prompt)
        with self._lock:
            self._sessions[sid] = sess
            running = sum(1 for s in self._sessions.values() if s.status is BackgroundStatus.RUNNING)
            if running >= self._max:
                # Stay PENDING; user can poll. The manager runs simple FIFO.
                return sess
            self._start_locked(sess)
        return sess

    def _start_locked(self, sess: BackgroundSession) -> None:
        """Transition ``sess`` to RUNNING and spin up its worker thread.

        Caller must hold ``self._lock``.
        """
        sess.status = BackgroundStatus.RUNNING
        sess.started_at = time.time()
        t = threading.Thread(target=self._run, args=(sess,), name=f"xerxes-bg-{sess.id}", daemon=True)
        self._threads[sess.id] = t
        t.start()

    def _run(self, sess: BackgroundSession) -> None:
        """Execute ``runner(sess)`` on the worker thread and record the outcome."""
        try:
            result = self._runner(sess)
            sess.result = result
            sess.status = BackgroundStatus.SUCCEEDED
        except Exception as exc:
            sess.error = f"{type(exc).__name__}: {exc}"
            sess.status = BackgroundStatus.FAILED
        finally:
            sess.finished_at = time.time()
            self._drain_pending()

    def _drain_pending(self) -> None:
        """Promote one ``PENDING`` session to ``RUNNING`` if capacity allows."""
        with self._lock:
            running = sum(1 for s in self._sessions.values() if s.status is BackgroundStatus.RUNNING)
            if running >= self._max:
                return
            for sess in self._sessions.values():
                if sess.status is BackgroundStatus.PENDING:
                    self._start_locked(sess)
                    return

    def get(self, session_id: str) -> BackgroundSession | None:
        """Return the session with ``session_id``, or ``None`` if unknown."""
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self) -> list[BackgroundSession]:
        """Return every tracked session in insertion order."""
        with self._lock:
            return list(self._sessions.values())

    def cancel(self, session_id: str) -> bool:
        """Cooperatively mark a session as cancelled.

        Worker threads are never killed; they finish their own work and only
        the bookkeeping changes. Returns ``True`` when a still-pending or
        running session was transitioned.
        """
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None or sess.status not in (BackgroundStatus.PENDING, BackgroundStatus.RUNNING):
                return False
            sess.status = BackgroundStatus.CANCELLED
            sess.finished_at = time.time()
            return True

    def shutdown(self, *, timeout: float = 5.0) -> None:
        """Stop accepting new sessions and join outstanding workers.

        After this returns, :meth:`submit` raises ``RuntimeError``. Running
        threads are joined for up to ``timeout`` seconds in aggregate.
        """
        self._stop.set()
        ends = time.time() + timeout
        with self._lock:
            threads = list(self._threads.values())
        for t in threads:
            remaining = max(0.0, ends - time.time())
            t.join(timeout=remaining)


__all__ = ["BackgroundSession", "BackgroundSessionManager", "BackgroundStatus", "RunFn"]
