# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.
"""Filesystem path resolution for Xerxes user data.

Centralises the location of Xerxes's per-user state directory (sessions,
skills, profiles, daemon files, agents) so it can be redirected via the
``XERXES_HOME`` environment variable. This is essential for:

- Containerised deployments (override to a mounted volume).
- Multi-tenant hosts (per-tenant home directories).
- Test isolation (per-test temporary directory).
- XDG-compliant setups (``XERXES_HOME=$XDG_DATA_HOME/xerxes``).

When ``XERXES_HOME`` is unset, the default ``~/.xerxes`` is used.
"""

from __future__ import annotations

import os
from pathlib import Path

XERXES_HOME_ENV = "XERXES_HOME"
_DEFAULT_DIR_NAME = ".xerxes"


def xerxes_home() -> Path:
    """Return the root Xerxes data directory.

    Resolution order:

    1. ``$XERXES_HOME`` environment variable (if set and non-empty).
    2. ``~/.xerxes`` (user home directory).

    The returned path is **not** created on disk; callers should
    ``mkdir(parents=True, exist_ok=True)`` themselves when writing.

    Returns:
        The :class:`~pathlib.Path` to the Xerxes data root.
    """
    override = os.environ.get(XERXES_HOME_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / _DEFAULT_DIR_NAME


def xerxes_subdir(*parts: str) -> Path:
    """Return a subdirectory under the Xerxes home.

    Convenience wrapper for ``xerxes_home() / part1 / part2 / ...``.
    The directory is **not** created.

    Args:
        *parts: Path segments to join under the Xerxes home.

    Returns:
        Joined :class:`~pathlib.Path` underneath the Xerxes home.
    """
    return xerxes_home().joinpath(*parts)
