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
"""Daemon source fingerprint shared by the server and TUI client."""

from __future__ import annotations

import hashlib
from pathlib import Path

DAEMON_PROTOCOL_VERSION = 35

_FINGERPRINT_FILES = (
    "daemon/fingerprint.py",
    "daemon/server.py",
    "daemon/runtime.py",
    "daemon/slash_commands.py",
    "streaming/loop.py",
    "context/window_usage.py",
)


def _compute_daemon_build_id() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in _FINGERPRINT_FILES:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()[:16]


DAEMON_BUILD_ID = _compute_daemon_build_id()


def daemon_build_id() -> str:
    """Return the daemon source fingerprint captured when this process imported it."""
    return DAEMON_BUILD_ID
