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
"""Per-platform session reset policies.

Supports three policies:

    * ``timeout``   — reset after N minutes of inactivity.
    * ``msg_count`` — reset after the session reaches N messages.
    * ``manual``    — only when the user explicitly asks (``/new``, ``/reset``).

``SessionResetPolicy`` and ``should_reset`` are pure data + pure function
so adapters can call them from any thread without coordination.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


class ResetTrigger(enum.Enum):
    """Which rule decides when a chat session should be reset.

    Attributes:
        TIMEOUT: Reset after a configurable idle period.
        MSG_COUNT: Reset once the session crosses a message threshold.
        MANUAL: Only the user can request a reset; never auto-reset.
    """

    TIMEOUT = "timeout"
    MSG_COUNT = "msg_count"
    MANUAL = "manual"


@dataclass
class SessionResetPolicy:
    """Configuration for the session-reset policy of a chat surface.

    Attributes:
        trigger: Which rule applies. Defaults to ``MANUAL``.
        timeout_minutes: Idle minutes before reset under ``TIMEOUT``.
        msg_count: Message-count ceiling under ``MSG_COUNT``.
    """

    trigger: ResetTrigger = ResetTrigger.MANUAL
    timeout_minutes: int = 60
    msg_count: int = 50


def should_reset(
    policy: SessionResetPolicy,
    *,
    last_message_at: datetime | None,
    message_count: int,
    manual_request: bool = False,
    now: datetime | None = None,
) -> bool:
    """Decide whether the session should reset right now.

    A ``manual_request`` always wins regardless of policy. For ``TIMEOUT``,
    naive ``last_message_at`` values are treated as UTC. ``MSG_COUNT`` is
    inclusive: at exactly ``policy.msg_count`` messages the session resets.

    Args:
        policy: The active reset policy.
        last_message_at: Timestamp of the most recent message in the
            session, or ``None`` if the session is empty.
        message_count: Number of messages currently in the session.
        manual_request: Set when the user just issued ``/new`` or ``/reset``.
        now: Optional override for "current time"; defaults to ``datetime.now(UTC)``.

    Returns:
        ``True`` when the session should be reset, ``False`` to keep it.
    """
    if manual_request:
        return True
    if policy.trigger is ResetTrigger.MANUAL:
        return False
    if policy.trigger is ResetTrigger.MSG_COUNT:
        return message_count >= policy.msg_count
    if policy.trigger is ResetTrigger.TIMEOUT:
        if last_message_at is None:
            return False
        if last_message_at.tzinfo is None:
            last_message_at = last_message_at.replace(tzinfo=UTC)
        threshold = (now or datetime.now(UTC)) - timedelta(minutes=policy.timeout_minutes)
        return last_message_at < threshold
    return False


__all__ = ["ResetTrigger", "SessionResetPolicy", "should_reset"]
