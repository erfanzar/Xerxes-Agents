# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
"""Thread-local active-session pointer for tools that need session state.

Most tools are stateless — they receive their inputs and produce an output.
A few need to reach back into the live :class:`xerxes.daemon.runtime.DaemonSession`
that owns the current turn: the async sub-agent orchestration tools
(``AwaitAgents``, ``ResetAgent``) need access to ``pending_steers`` and the
cancel flag so the main agent can sleep until either sub-agents finish or
the user types something to wake it.

The streaming loop runs each turn on a dedicated worker thread (one per
session), so a :class:`threading.local` is the natural carrier. The daemon's
``TurnRunner`` sets the active session before driving the loop and clears it
in the ``finally`` block.
"""

from __future__ import annotations

import threading
from typing import Any

_local = threading.local()


def set_active_session(session: Any | None) -> None:
    """Bind ``session`` to the current thread for tool access; ``None`` clears it."""
    _local.session = session


def get_active_session() -> Any | None:
    """Return the session bound to the current thread, or ``None``."""
    return getattr(_local, "session", None)


__all__ = ["get_active_session", "set_active_session"]
