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
"""ACP registry metadata + helper to write ``agent.json``.

Compatible IDE clients discover Xerxes via this manifest. The
canonical location is ``$XDG_CONFIG_HOME/agent-registry/`` or
``~/.config/agent-registry/``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

REGISTRY_METADATA: dict[str, Any] = {
    "name": "xerxes",
    "display_name": "Xerxes",
    "description": "Multi-agent coding assistant built on Xerxes-Agents.",
    "version": "0.2.0",
    "vendor": "Erfan Zare Chavoshi",
    "license": "Apache-2.0",
    "homepage": "https://github.com/erfanzar/Xerxes-Agents",
    "distribution": {
        "type": "command",
        "command": "xerxes-acp",
        "args": [],
    },
    "capabilities": {
        "streaming": True,
        "tools": True,
        "permissions": True,
        "sessions": True,
        "fork": True,
        "cancel": True,
        "models": True,
    },
}


def _default_registry_dir() -> Path:
    """Return ``$XDG_CONFIG_HOME/agent-registry`` (or ``~/.config/...`` fallback)."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "agent-registry"


def write_registry_file(target_dir: Path | None = None) -> Path:
    """Write ``agent.json`` to the user's ACP registry directory.

    Returns the path written. Creates the directory if missing."""
    target = (target_dir or _default_registry_dir()) / "xerxes"
    target.mkdir(parents=True, exist_ok=True)
    out = target / "agent.json"
    out.write_text(json.dumps(REGISTRY_METADATA, indent=2), encoding="utf-8")
    return out


__all__ = ["REGISTRY_METADATA", "write_registry_file"]
