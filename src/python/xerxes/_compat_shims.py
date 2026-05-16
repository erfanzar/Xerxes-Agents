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
"""Lightweight path helpers that avoid importing ``xerxes.core.paths``.

Used by modules that participate in circular imports with the core
paths module, or by plugin entry points where pulling the full path
machinery would be too heavy. Mirrors the subset of behaviour needed
by callers without the full configuration surface.
"""

from __future__ import annotations

import os
from pathlib import Path


def xerxes_subdir_safe(*parts: str) -> Path:
    """Resolve a path under ``$XERXES_HOME`` (default ``~/.xerxes``).

    Does not create the directory or read any other configuration; the
    caller is responsible for ``mkdir`` if persistence is required.
    """
    base = os.environ.get("XERXES_HOME") or str(Path.home() / ".xerxes")
    return Path(base, *parts)


__all__ = ["xerxes_subdir_safe"]
