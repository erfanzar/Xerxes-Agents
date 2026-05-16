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
"""Context-utilization bar.

Render a filled/empty block meter that visualises ``used / window``
token utilization for the footer."""

from __future__ import annotations

FILLED_BLOCK = "█"
HALF_BLOCK = "▌"
EMPTY_BLOCK = "░"


def context_bar(
    *,
    used: int,
    window: int,
    width: int = 24,
    filled: str = FILLED_BLOCK,
    half: str = HALF_BLOCK,
    empty: str = EMPTY_BLOCK,
) -> str:
    """Return a ``width``-character meter.

    Halves a block when utilization falls between integer fill counts
    so very low usage still produces a visible mark."""
    if width <= 0:
        return ""
    window = max(1, int(window))
    used = max(0, int(used))
    cells = (used / window) * width if used < window else width
    full = int(cells)
    has_half = (cells - full) >= 0.5
    if full > width:
        full = width
        has_half = False
    extra = 1 if has_half else 0
    return (filled * full) + (half if has_half else "") + (empty * (width - full - extra))


def context_bar_with_pct(
    *,
    used: int,
    window: int,
    width: int = 24,
    show_pct: bool = True,
) -> str:
    """Return :func:`context_bar` plus a right-aligned ``NN.N%`` suffix.

    Set ``show_pct=False`` to suppress the percentage and get the raw bar."""
    bar = context_bar(used=used, window=window, width=width)
    if not show_pct:
        return bar
    pct = (used / window * 100) if window > 0 else 0
    return f"{bar} {pct:5.1f}%"


__all__ = ["EMPTY_BLOCK", "FILLED_BLOCK", "HALF_BLOCK", "context_bar", "context_bar_with_pct"]
