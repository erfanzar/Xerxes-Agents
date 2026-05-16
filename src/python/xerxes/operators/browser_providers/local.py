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
"""Default in-process provider that delegates to :class:`BrowserManager`."""

from __future__ import annotations

import importlib.util
from typing import Any

from . import BrowserProvider, BrowserSession


class LocalProvider(BrowserProvider):
    """Marker provider for the in-process Playwright path.

    :class:`BrowserManager` already drives Playwright directly, so this
    provider simply verifies that Playwright is importable and returns a
    sentinel :class:`BrowserSession`. Page lifecycle stays with the
    surrounding manager.
    """

    name = "local"

    def open(self, *, headless: bool = True, **kwargs: Any) -> BrowserSession:
        """Verify Playwright is installed and return a sentinel session.

        Raises:
            RuntimeError: When the ``playwright`` package is missing.
        """
        if importlib.util.find_spec("playwright") is None:
            raise RuntimeError("playwright not installed; `pip install playwright && playwright install`")
        # Delegate to the existing BrowserManager construction. The actual
        # page lifecycle is owned by that manager; here we just hand back
        # a marker so the operator chooses the local path.
        return BrowserSession(provider=self.name, metadata={"headless": headless})

    def close(self, session: BrowserSession) -> None:
        """No-op: the in-process page is owned by :class:`BrowserManager`."""
        return None
