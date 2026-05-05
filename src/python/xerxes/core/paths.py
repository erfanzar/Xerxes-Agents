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
"""Filesystem path helpers for Xerxes.

Provides ``xerxes_home()`` and ``xerxes_subdir()`` for resolving the project's
home directory (default ``~/.xerxes``, overridable via ``XERXES_HOME``).
"""

from __future__ import annotations

import os
from pathlib import Path

XERXES_HOME_ENV = "XERXES_HOME"
_DEFAULT_DIR_NAME = ".xerxes"


def xerxes_home() -> Path:
    """Return the Xerxes home directory path.

    Uses ``XERXES_HOME`` environment variable when set, otherwise
    ``~/.xerxes``.

    Returns:
        Path: OUT: resolved home directory.
    """
    override = os.environ.get(XERXES_HOME_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / _DEFAULT_DIR_NAME


def xerxes_subdir(*parts: str) -> Path:
    """Return a subpath inside the Xerxes home directory.

    Args:
        *parts (str): IN: path components to join.

    Returns:
        Path: OUT: resolved subpath.
    """
    return xerxes_home().joinpath(*parts)
