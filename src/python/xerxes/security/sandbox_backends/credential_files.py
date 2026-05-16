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
"""Registry of credential files that should be pushed to remote sandboxes.

Agents register paths (``~/.aws/credentials``, ``~/.ssh/id_ed25519``,
etc.) that are allow-listed for transfer to Modal/SSH/Daytona sandboxes.
Anything not registered is never sent.

The registry is process-global so any tool can opt in to including
a credential file, but everything goes through ``allowed_paths``
before transfer."""

from __future__ import annotations

import os
import threading
from pathlib import Path

_ALLOWLIST: set[Path] = set()
_LOCK = threading.Lock()


def register(path: str | Path) -> Path:
    """Add ``path`` to the allow-list. Returns the resolved Path."""
    resolved = Path(path).expanduser().resolve()
    with _LOCK:
        _ALLOWLIST.add(resolved)
    return resolved


def unregister(path: str | Path) -> bool:
    """Remove ``path`` from the allow-list; return True if it was present."""
    resolved = Path(path).expanduser().resolve()
    with _LOCK:
        if resolved in _ALLOWLIST:
            _ALLOWLIST.remove(resolved)
            return True
        return False


def allowed_paths() -> list[Path]:
    """Snapshot of currently-registered credential paths, sorted."""
    with _LOCK:
        return sorted(_ALLOWLIST)


def is_allowed(path: str | Path) -> bool:
    """Return True if ``path`` (after expanduser+resolve) is allow-listed."""
    resolved = Path(path).expanduser().resolve()
    with _LOCK:
        return resolved in _ALLOWLIST


def clear() -> None:
    """Drop every registered path (used by tests to isolate state)."""
    with _LOCK:
        _ALLOWLIST.clear()


def env_passthrough(names: list[str]) -> dict[str, str]:
    """Return a filtered ``os.environ`` containing only ``names``.

    Convenience for the sandbox backends to surface a small, allow-listed
    subset of the parent env into the sandbox."""
    return {name: os.environ[name] for name in names if name in os.environ}


__all__ = [
    "allowed_paths",
    "clear",
    "env_passthrough",
    "is_allowed",
    "register",
    "unregister",
]
