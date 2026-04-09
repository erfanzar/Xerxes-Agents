# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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

"""Global runtime config context.

Stores the active runtime config (model, base_url, api_key, sampling params)
so that sub-agents and tools can inherit provider settings without explicit
threading through every call site.

Usage::

    from calute.runtime.config_context import set_config, get_config

    # Set once at startup (bridge server does this).
    set_config({"model": "qwen3-8.19b", "base_url": "http://...", "api_key": "sk-..."})

    # Read anywhere (AgentTool, sub-agents, etc.).
    cfg = get_config()
    base_url = cfg.get("base_url", "")
"""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_config: dict[str, Any] = {}

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
}


def set_config(config: dict[str, Any]) -> None:
    """Set the global runtime config. Called by the bridge server."""
    global _config
    with _lock:
        _config = dict(config)


def get_config() -> dict[str, Any]:
    """Get a copy of the global runtime config."""
    with _lock:
        return dict(_config)


def get_inheritable() -> dict[str, Any]:
    """Get only the inheritable keys (provider + sampling) for sub-agents."""
    with _lock:
        return {k: v for k, v in _config.items() if k in _INHERITABLE and v is not None and v != ""}
