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
"""Resolve the Xerxes home directory and subpaths consistently.

Every persisted resource — sessions, profiles, skills, daemon sockets, agent
workspaces — lives under :func:`xerxes_home`. The default is ``~/.xerxes``;
setting the ``XERXES_HOME`` environment variable redirects everything to an
explicit location (useful for tests and sandboxes). All callers should route
their paths through :func:`xerxes_subdir` rather than rebuilding the prefix
by hand.
"""

from __future__ import annotations

import os
from pathlib import Path

XERXES_HOME_ENV = "XERXES_HOME"
_DEFAULT_DIR_NAME = ".xerxes"


def xerxes_home() -> Path:
    """Return the Xerxes home directory (``$XERXES_HOME`` or ``~/.xerxes``)."""
    override = os.environ.get(XERXES_HOME_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / _DEFAULT_DIR_NAME


def xerxes_subdir(*parts: str) -> Path:
    """Join ``parts`` under :func:`xerxes_home` without creating the directory."""
    return xerxes_home().joinpath(*parts)
