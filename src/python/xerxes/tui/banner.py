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
"""Welcome banner with compact mode for narrow terminals.

``render_banner`` picks the full or compact layout based on terminal width."""

from __future__ import annotations

import shutil
from dataclasses import dataclass

FULL_LOGO = """\
 ██╗  ██╗███████╗██████╗ ██╗  ██╗███████╗███████╗
 ╚██╗██╔╝██╔════╝██╔══██╗╚██╗██╔╝██╔════╝██╔════╝
  ╚███╔╝ █████╗  ██████╔╝ ╚███╔╝ █████╗  ███████╗
  ██╔██╗ ██╔══╝  ██╔══██╗ ██╔██╗ ██╔══╝  ╚════██║
 ██╔╝ ██╗███████╗██║  ██║██╔╝ ██╗███████╗███████║
 ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝
"""

COMPACT_LOGO = "✦ Xerxes ✦"


@dataclass
class BannerData:
    """Data the banner renders alongside the logo.

    Attributes:
        version: Currently installed Xerxes version.
        model: Active model identifier; falls back to a "(unset)" hint.
        workspace: Workspace label or path; "(none)" when unset.
        session_id: Daemon-assigned session id; "(new)" before init.
        tip: Optional tip-of-the-day line shown beneath the box.
        update_available: Newer version string, or "" when up to date.
    """

    version: str = "0.2.0"
    model: str = ""
    workspace: str = ""
    session_id: str = ""
    tip: str = ""
    update_available: str = ""  # "" or e.g. "0.3.0"


def _terminal_width(default: int = 100) -> int:
    """Return the terminal width, defaulting to ``default`` when unavailable."""
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except OSError:
        return default


def render_banner(data: BannerData, *, terminal_width: int | None = None) -> str:
    """Return banner text suitable for direct print().

    Full layout: full ASCII logo + session info box + tip.
    Compact layout (terminals < 64 cols): one-line logo + key info."""
    width = terminal_width if terminal_width is not None else _terminal_width()
    if width < 64:
        return _compact(data)
    return _full(data)


def _full(data: BannerData) -> str:
    """Render the full ASCII banner with session info box and optional tip."""
    box_width = max(64, len(FULL_LOGO.splitlines()[0]))
    info_lines = [
        f"Xerxes (v{data.version}){'  ↑ update ' + data.update_available if data.update_available else ''}",
        f"model:     {data.model or '(unset — run /provider)'}",
        f"workspace: {data.workspace or '(none)'}",
        f"session:   {data.session_id or '(new)'}",
    ]
    box = []
    inner_w = max(48, box_width - 4)
    box.append("╭" + "─" * (inner_w + 2) + "╮")
    for line in info_lines:
        clipped = line if len(line) <= inner_w else line[: inner_w - 1] + "…"
        box.append("│ " + clipped.ljust(inner_w) + " │")
    box.append("╰" + "─" * (inner_w + 2) + "╯")
    pieces = [FULL_LOGO, "\n".join(box)]
    if data.tip:
        pieces.append(f"\n💡 {data.tip}")
    return "\n".join(pieces)


def _compact(data: BannerData) -> str:
    """Render the narrow-terminal variant: one-line logo + a 1-2 info lines."""
    line1 = f"{COMPACT_LOGO}  v{data.version}"
    if data.update_available:
        line1 += f"  (↑{data.update_available})"
    line2_parts = []
    if data.model:
        line2_parts.append(f"model={data.model}")
    if data.session_id:
        line2_parts.append(f"session={data.session_id[:8]}")
    if data.workspace:
        line2_parts.append(f"ws={data.workspace}")
    line2 = " ".join(line2_parts)
    pieces = [line1]
    if line2:
        pieces.append(line2)
    if data.tip:
        # Truncate so we never push width past 70 cols.
        tip = data.tip if len(data.tip) <= 64 else data.tip[:63] + "…"
        pieces.append(f"💡 {tip}")
    return "\n".join(pieces)


__all__ = ["COMPACT_LOGO", "FULL_LOGO", "BannerData", "render_banner"]
