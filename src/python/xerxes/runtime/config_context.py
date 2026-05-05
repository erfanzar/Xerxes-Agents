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
"""Config context module for Xerxes.

Exports:
    - set_config
    - get_config
    - get_inheritable
    - set_event_callback
    - get_event_callback
    - emit_event"""

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
    """Set the config.

    Args:
        config (dict[str, Any]): IN: config. OUT: Consumed during execution."""

    global _config
    with _lock:
        _config = dict(config)


def get_config() -> dict[str, Any]:
    """Retrieve the config.

    Returns:
        dict[str, Any]: OUT: Result of the operation."""

    with _lock:
        return dict(_config)


def get_inheritable() -> dict[str, Any]:
    """Retrieve the inheritable.

    Returns:
        dict[str, Any]: OUT: Result of the operation."""

    with _lock:
        return {k: v for k, v in _config.items() if k in _INHERITABLE and v is not None and v != ""}


def set_event_callback(cb: Callable[[str, dict[str, Any]], None] | None) -> None:
    """Set the event callback.

    Args:
        cb (Callable[[str, dict[str, Any]], None] | None): IN: cb. OUT: Consumed during execution."""

    global _event_callback
    with _lock:
        _event_callback = cb


def get_event_callback() -> Callable[[str, dict[str, Any]], None] | None:
    """Retrieve the event callback.

    Returns:
        Callable[[str, dict[str, Any]], None] | None: OUT: Result of the operation."""

    with _lock:
        return _event_callback


def emit_event(event_type: str, data: dict[str, Any]) -> None:
    """Emit event.

    Args:
        event_type (str): IN: event type. OUT: Consumed during execution.
        data (dict[str, Any]): IN: data. OUT: Consumed during execution."""

    cb = get_event_callback()
    if cb is not None:
        try:
            cb(event_type, data)
        except Exception:
            pass
