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
"""Per-thread cooperative interrupt signalling.

Tools running inside the agent loop occasionally need to learn that the user
pressed Ctrl+C (or sent ``/cancel``) so they can stop mid-flight.
:class:`InterruptToken` is the cooperative primitive long-running tools poll
between work units; module-level helpers expose the active token via a
``threading.local`` so deep call stacks (and ``with`` blocks) can read it
without explicit plumbing.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class InterruptToken:
    """Thread-safe, reusable interrupt flag built on ``threading.Event``.

    Call :meth:`set` from any thread to request interruption; cooperating
    tool code calls :meth:`is_set` (or :meth:`raise_if_set`) at checkpoints
    and bails out cleanly when the flag is up. :meth:`clear` resets the flag
    so the same instance can be reused across runs.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        """Create an unset token backed by a fresh ``threading.Event``."""
        self._event = threading.Event()

    def set(self) -> None:
        """Raise the interrupt flag; safe to call from any thread."""
        self._event.set()

    def clear(self) -> None:
        """Lower the interrupt flag so the token can be reused."""
        self._event.clear()

    def is_set(self) -> bool:
        """Return whether interruption has been requested."""
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the flag is set or ``timeout`` elapses."""
        return self._event.wait(timeout)

    def raise_if_set(self) -> None:
        """Raise :class:`KeyboardInterrupt` when the flag is set.

        Convenience for tools that prefer to fail fast at natural checkpoints
        instead of branching on :meth:`is_set` manually.
        """
        if self._event.is_set():
            raise KeyboardInterrupt("Tool interrupted by user")


_thread_local = threading.local()


def current_token() -> InterruptToken | None:
    """Return the :class:`InterruptToken` installed on this thread, or ``None``."""
    return getattr(_thread_local, "token", None)


def set_current_token(token: InterruptToken | None) -> None:
    """Install ``token`` as the current thread's interrupt token."""
    _thread_local.token = token


def clear_current_token() -> None:
    """Remove any token currently installed on this thread."""
    set_current_token(None)


@contextmanager
def interrupt_scope(token: InterruptToken | None = None) -> Iterator[InterruptToken]:
    """Install an interrupt token for the duration of a ``with`` block.

    Args:
        token: Token to install. A fresh one is created when ``None``.

    Yields:
        The installed token so callers can hand it to threads they spawn.

    Note:
        The previous thread-local token (if any) is restored on exit, so
        nested scopes nest correctly.
    """

    prev = current_token()
    tok = token if token is not None else InterruptToken()
    set_current_token(tok)
    try:
        yield tok
    finally:
        set_current_token(prev)


__all__ = [
    "InterruptToken",
    "clear_current_token",
    "current_token",
    "interrupt_scope",
    "set_current_token",
]
