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
"""External memory provider plugins.

Each subdirectory is a plugin shipping a ``MemoryProvider`` subclass
gated by an optional dependency. Importing this package eagerly
registers every plugin whose import succeeds; plugins missing their
upstream SDK silently skip registration (their ``is_available()``
returns False)."""

from __future__ import annotations

import importlib
import logging

from ..provider import register

logger = logging.getLogger(__name__)

_PLUGIN_NAMES = (
    "honcho",
    "mem0",
    "hindsight",
    "holographic",
    "retaindb",
    "openviking",
    "supermemory",
    "byterover",
)


def _try_load(name: str) -> None:
    """Import the plugin module and register its ``PROVIDER`` singleton; swallow failures."""
    module_path = f"{__name__}.{name}"
    try:
        module = importlib.import_module(module_path)
        provider = module.PROVIDER  # type: ignore[attr-defined]
        register(provider)
    except Exception as exc:
        logger.debug("memory plugin %r unavailable: %s", name, exc)


for _name in _PLUGIN_NAMES:
    _try_load(_name)


__all__: list[str] = []
