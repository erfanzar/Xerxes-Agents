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
"""Process-global runtime config snapshot and event bus.

Stores the active CLI/daemon configuration as a lock-protected dict that
modules (sub-agent spawners, tool wrappers, telemetry) can consult without
plumbing it through every call. :data:`_INHERITABLE` lists the keys that
sub-agents should pick up from their parent. A single optional event callback
funnels lightweight events (sub-agent start/stop, model change, etc.) into
whichever bridge or telemetry sink the host installs.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

_lock = threading.Lock()
_config: dict[str, Any] = {}
_event_callback: Callable[[str, dict[str, Any]], None] | None = None

_INHERITABLE = {
    "model",
    "base_url",
    "api_key",
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "min_p",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "permission_mode",
}


def set_config(config: dict[str, Any]) -> None:
    """Replace the process-global config snapshot with a copy of ``config``."""

    global _config
    with _lock:
        _config = dict(config)


def get_config() -> dict[str, Any]:
    """Return a shallow copy of the current global config."""

    with _lock:
        return dict(_config)


def get_inheritable() -> dict[str, Any]:
    """Return only the config keys that sub-agents should inherit.

    Filters :func:`get_config` to the whitelist in :data:`_INHERITABLE`,
    skipping ``None`` and empty-string values so callers can ``**spread``
    the result without clobbering child defaults.
    """

    with _lock:
        return {k: v for k, v in _config.items() if k in _INHERITABLE and v is not None and v != ""}


def set_event_callback(cb: Callable[[str, dict[str, Any]], None] | None) -> None:
    """Install (or clear with ``None``) the global event-bus callback."""

    global _event_callback
    with _lock:
        _event_callback = cb


def get_event_callback() -> Callable[[str, dict[str, Any]], None] | None:
    """Return the currently installed event callback, or ``None``."""

    with _lock:
        return _event_callback


def emit_event(event_type: str, data: dict[str, Any]) -> None:
    """Forward ``(event_type, data)`` to the installed callback, swallowing errors.

    Exceptions raised by the callback are intentionally suppressed so a
    misbehaving telemetry sink can't crash the agent loop.
    """

    cb = get_event_callback()
    if cb is not None:
        try:
            cb(event_type, data)
        except Exception:
            pass
