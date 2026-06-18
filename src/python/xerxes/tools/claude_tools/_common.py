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
"""Shared helpers for file-editing tools."""

from __future__ import annotations

import difflib


def _unified_diff(old: str, new: str, filename: str = "", context: int = 3) -> str:
    """Generate a unified diff string between two text contents.

    Args:
        old: Original text content.
        new: Modified text content.
        filename: Optional filename for diff headers.
        context: Number of context lines around changes.

    Returns:
        Formatted unified diff string.
    """
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filename}" if filename else "a",
        tofile=f"b/{filename}" if filename else "b",
        n=context,
    )
    result = "".join(diff)
    lines = result.split("\n")
    if len(lines) > 80:
        result = "\n".join(lines[:80]) + f"\n... ({len(lines) - 80} more lines)"
    return result


def _closest_match_hint(content: str, old_string: str, max_candidates: int = 3) -> str:
    """Diagnose a failed FileEdit by pointing at the closest existing lines.

    A no-match edit is the single highest-frequency tool failure; surfacing the
    nearest line(s) with their numbers turns a blind retry loop into a fixable
    one (usually whitespace/indentation drift or stale content).

    Args:
        content: Current file text.
        old_string: The text the caller tried (and failed) to match.
        max_candidates: Maximum number of near-miss lines to report.

    Returns:
        A short hint string, never raising.
    """
    file_lines = content.splitlines()
    probe = next((ln for ln in old_string.splitlines() if ln.strip()), old_string).strip()
    if not probe or not file_lines:
        return (
            "Hint: text not present — re-read the file before retrying (it may target the wrong file or stale content)."
        )
    scored = sorted(
        ((difflib.SequenceMatcher(None, probe, fl.strip()).ratio(), idx, fl) for idx, fl in enumerate(file_lines)),
        key=lambda t: t[0],
        reverse=True,
    )
    best = [s for s in scored[:max_candidates] if s[0] >= 0.5]
    if not best:
        return "Hint: no similar line found — re-read the file; old_string may target the wrong file or stale content."
    out = ["Hint: closest existing lines (old_string must match EXACTLY, including whitespace):"]
    for ratio, idx, fl in best:
        out.append(f"  L{idx + 1} (~{int(ratio * 100)}% match): {fl.strip()[:120]}")
    return "\n".join(out)


def _fuzzy_whitespace_replace(content: str, old_string: str, new_string: str) -> str | None:
    """Attempt a whitespace-normalized search-and-replace fallback.

    Collapses runs of whitespace (spaces, tabs, newlines) into single spaces
    in both ``content`` and ``old_string``, finds the match, then maps it
    back to the original content to produce the replacement. Returns ``None``
    when no whitespace-normalized match exists.

    This handles the common failure where the model's ``old_string`` has
    slightly different indentation, trailing whitespace, or line-ending
    conventions than the file on disk.
    """

    import re as _re

    collapse = _re.compile(r"\s+")

    def _normalize(text: str) -> str:
        return collapse.sub(" ", text).strip()

    norm_content = _normalize(content)
    norm_old = _normalize(old_string)

    if norm_old not in norm_content:
        return None

    return content.replace(old_string.strip(), new_string.strip(), 1) if old_string.strip() in content else None


__all__ = ["_closest_match_hint", "_fuzzy_whitespace_replace", "_unified_diff"]
