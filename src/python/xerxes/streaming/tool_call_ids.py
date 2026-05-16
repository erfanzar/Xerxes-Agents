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
"""Deterministic tool-call ID generation.

Tool-call IDs are derived from a SHA-256 hash of ``name + canonical(args)``
so a resumed session reproduces the same IDs as the original run. This is
what makes session replay, audit-log diffing, and idempotent dedup tractable.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonicalize_kwargs(kwargs: dict[str, Any]) -> str:
    """JSON-serialize kwargs with sorted keys for stable hashing.

    Falls back to ``str(value)`` for non-serializable inputs so the
    function never raises — at the cost of slightly weaker stability
    for exotic argument types."""

    try:
        return json.dumps(kwargs, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(sorted(kwargs.items()))


def deterministic_tool_call_id(
    name: str,
    kwargs: dict[str, Any],
    *,
    prefix: str = "call_",
    length: int = 16,
) -> str:
    """Return a stable ID for ``(name, kwargs)``.

    The same name + kwargs produces the same ID across runs and
    processes. Use ``prefix`` to namespace IDs from different sources
    (e.g. ``call_`` matches OpenAI's convention)."""

    payload = f"{name}|{canonicalize_kwargs(kwargs)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:length]}"


__all__ = ["canonicalize_kwargs", "deterministic_tool_call_id"]
