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
"""Generic single-select panel state + countdown.

Covers selection navigation, the approval timer, and the five-option
approval surface."""

from __future__ import annotations

import enum
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field


class ApprovalChoice(enum.Enum):
    """Discrete responses a user can pick on an approval panel.

    ``APPROVE_ONCE`` covers a single tool call; ``APPROVE_FOR_SESSION``
    persists for the rest of the current session; ``APPROVE_ALWAYS``
    writes to the on-disk approvals store; ``VIEW`` is a peek (e.g. show
    a diff) that leaves the request pending."""

    APPROVE = "approve"
    APPROVE_ONCE = "approve_once"
    APPROVE_FOR_SESSION = "approve_for_session"
    APPROVE_ALWAYS = "approve_always"
    DENY = "deny"
    VIEW = "view"


# Default 5-option approval layout (with a fallback when ``view`` is omitted).
DEFAULT_APPROVAL_OPTIONS: tuple[ApprovalChoice, ...] = (
    ApprovalChoice.APPROVE,
    ApprovalChoice.APPROVE_ONCE,
    ApprovalChoice.APPROVE_FOR_SESSION,
    ApprovalChoice.APPROVE_ALWAYS,
    ApprovalChoice.DENY,
)


@dataclass
class PanelSelection:
    """Track which option a modal panel currently highlights.

    Used by approval / clarify / model-picker panels.

    Attributes:
        options: Ordered list of selectable labels.
        index: Currently highlighted index (wraps with ``move``).
    """

    options: list[str]
    index: int = 0

    def move(self, direction: int) -> int:
        """Shift ``index`` by ``direction`` with wrap-around. Returns the new index."""
        if not self.options:
            return 0
        self.index = (self.index + direction) % len(self.options)
        return self.index

    def up(self) -> int:
        """Move selection one step toward the start of the list."""
        return self.move(-1)

    def down(self) -> int:
        """Move selection one step toward the end of the list."""
        return self.move(1)

    def set(self, idx: int) -> int:
        """Jump to absolute index ``idx`` (modulo list length)."""
        if not self.options:
            return 0
        self.index = idx % len(self.options)
        return self.index

    def selected(self) -> str:
        """Return the highlighted label, or ``""`` if there are no options."""
        if not self.options:
            return ""
        return self.options[self.index]


@dataclass
class ApprovalPanelState:
    """Selection state over the 5-option approval panel.

    Attributes:
        choices: Ordered tuple of :class:`ApprovalChoice` values shown
            in the panel. Defaults to :data:`DEFAULT_APPROVAL_OPTIONS`.
        selection: Underlying :class:`PanelSelection` populated after
            ``__post_init__`` from the choice values.
    """

    choices: tuple[ApprovalChoice, ...] = field(default=DEFAULT_APPROVAL_OPTIONS)
    selection: PanelSelection = field(init=False)

    def __post_init__(self) -> None:
        """Materialize :attr:`selection` from the configured ``choices``."""
        self.selection = PanelSelection(options=[c.value for c in self.choices])

    def up(self) -> str:
        """Move the cursor up and return the newly highlighted choice value."""
        self.selection.up()
        return self.selection.selected()

    def down(self) -> str:
        """Move the cursor down and return the newly highlighted choice value."""
        self.selection.down()
        return self.selection.selected()

    def current(self) -> ApprovalChoice:
        """Return the highlighted choice as an :class:`ApprovalChoice` enum."""
        return ApprovalChoice(self.selection.selected())


@dataclass
class ApprovalCountdown:
    """Time-out timer for an approval panel.

    The TUI calls ``start(callback)`` when a permission request appears;
    when the deadline elapses, the callback fires with the default
    action (typically deny). ``cancel`` stops the timer on user input."""

    timeout_seconds: float = 60.0
    _start_time: float = field(default=0.0, init=False)
    _timer: threading.Timer | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def start(self, on_timeout: Callable[[], None]) -> None:
        """Arm a fresh daemon timer; fires ``on_timeout`` after ``timeout_seconds``.

        Any in-flight timer is cancelled first, so callers can safely
        restart on key press without leaking threads."""
        with self._lock:
            self._cancel_locked()
            self._start_time = time.monotonic()
            t = threading.Timer(self.timeout_seconds, on_timeout)
            t.daemon = True
            self._timer = t
            t.start()

    def cancel(self) -> None:
        """Cancel the in-flight timer if any; safe to call multiple times."""
        with self._lock:
            self._cancel_locked()

    def _cancel_locked(self) -> None:
        """Inner cancel; must be called with ``self._lock`` held."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._start_time = 0.0

    def elapsed(self) -> float:
        """Return seconds since :meth:`start` (0 when no timer is active)."""
        with self._lock:
            if self._start_time == 0.0:
                return 0.0
            return time.monotonic() - self._start_time

    def remaining(self) -> float:
        """Return seconds left before the timeout fires; never negative."""
        with self._lock:
            if self._start_time == 0.0:
                return 0.0
            return max(0.0, self.timeout_seconds - (time.monotonic() - self._start_time))

    def is_active(self) -> bool:
        """Return ``True`` while the underlying threading.Timer is alive."""
        with self._lock:
            return self._timer is not None and self._timer.is_alive()


__all__ = [
    "DEFAULT_APPROVAL_OPTIONS",
    "ApprovalChoice",
    "ApprovalCountdown",
    "ApprovalPanelState",
    "PanelSelection",
]
