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
"""Camofox stealth Firefox provider.

Spawns a Node.js bridge subprocess (path supplied via the
``XERXES_CAMOFOX_SCRIPT`` environment variable) that exposes a CDP
endpoint :class:`BrowserManager` can attach to.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

from . import BrowserProvider, BrowserSession


class CamofoxProvider(BrowserProvider):
    """Provider that launches a local Camofox-bridge Node process."""

    name = "camofox"

    def open(self, *, headless: bool = True, **kwargs: Any) -> BrowserSession:
        """Start the Camofox bridge subprocess and return its session marker.

        Raises:
            RuntimeError: When ``node`` is not on ``PATH`` or the
                ``XERXES_CAMOFOX_SCRIPT`` env var is unset.
        """
        node = shutil.which("node")
        if node is None:
            raise RuntimeError("Node.js required for camofox stealth backend; install node first")
        script = os.environ.get("XERXES_CAMOFOX_SCRIPT")
        if not script:
            raise RuntimeError("Set XERXES_CAMOFOX_SCRIPT to the camofox-bridge JS entry point")
        # Spawn but do NOT wait — the bridge keeps running and exposes a CDP port.
        proc = subprocess.Popen([node, script, "--headless" if headless else "--no-headless"])
        return BrowserSession(provider=self.name, metadata={"pid": proc.pid, "script": script})

    def close(self, session: BrowserSession) -> None:
        """Send ``SIGTERM`` to the bridge process if it is still alive."""
        pid = (session.metadata or {}).get("pid")
        if pid:
            try:
                os.kill(int(pid), 15)  # SIGTERM
            except (ProcessLookupError, PermissionError):
                pass
