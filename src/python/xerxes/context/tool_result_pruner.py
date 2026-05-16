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
"""Heuristic pre-pruning of tool outputs before LLM compression.

Runs *before* any expensive LLM summarization pass during context
compaction. The goal is to cheaply shrink obviously oversized tool
outputs to informative placeholders so the LLM (and the next turn's
context) sees fewer tokens.

Examples of what gets pruned:

* A 50 000-line file read collapses to first/last N lines plus an
  ``[... K lines omitted ...]`` marker.
* A binary blob shrinks to a ``[N bytes of binary content elided]``
  placeholder.
* A wide single-line blob becomes head + tail chars with an omission
  marker for what was dropped.
"""

from __future__ import annotations

from typing import Any

DEFAULT_MAX_CHARS = 4_000  # keep up to this many chars verbatim
DEFAULT_HEAD_LINES = 40
DEFAULT_TAIL_LINES = 20


def _is_binary_blob(content: str) -> bool:
    """Heuristic: ``True`` when >30% of the first 1KB is non-printable."""
    if not content:
        return False
    sample = content[:1024]
    # Heuristic: more than 30% non-printable bytes → binary.
    non_print = sum(1 for c in sample if not (c.isprintable() or c in "\n\r\t"))
    return non_print > len(sample) * 0.3


def _truncate_text(content: str, head_lines: int, tail_lines: int, max_chars: int) -> str:
    """Return ``content`` truncated to head+tail lines or chars with a marker."""
    lines = content.splitlines()
    if len(lines) > head_lines + tail_lines:
        head = "\n".join(lines[:head_lines])
        tail = "\n".join(lines[-tail_lines:])
        omitted = len(lines) - head_lines - tail_lines
        return f"{head}\n\n[... {omitted} lines omitted by pre-pruning ...]\n\n{tail}"
    # Few lines but still too long (e.g. one very wide blob): char-truncate.
    head_chars = max(1, max_chars // 2)
    tail_chars = max(1, max_chars - head_chars)
    head_text = content[:head_chars]
    tail_text = content[-tail_chars:]
    omitted = len(content) - head_chars - tail_chars
    return f"{head_text}\n\n[... {omitted} chars omitted by pre-pruning ...]\n\n{tail_text}"


def prune_tool_result(
    content: Any,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    head_lines: int = DEFAULT_HEAD_LINES,
    tail_lines: int = DEFAULT_TAIL_LINES,
) -> tuple[Any, bool]:
    """Prune ``content`` heuristically; return ``(pruned, did_prune)``.

    Non-string content is returned unchanged with ``did_prune=False``;
    structured tool results are left for the LLM compressor to handle.
    Strings under ``max_chars`` pass through. Binary content becomes a
    byte-count placeholder. Otherwise the content is head+tail
    truncated with an omission marker.
    """

    if not isinstance(content, str):
        return content, False
    if len(content) <= max_chars:
        return content, False
    if _is_binary_blob(content):
        return f"[{len(content)} bytes of binary content elided by pre-pruning]", True
    return _truncate_text(content, head_lines, tail_lines, max_chars), True


def prune_messages(
    messages: list[dict[str, Any]],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    head_lines: int = DEFAULT_HEAD_LINES,
    tail_lines: int = DEFAULT_TAIL_LINES,
    protect_last: int = 4,
) -> tuple[list[dict[str, Any]], int]:
    """Return a new message list with oversized tool results pruned.

    The last ``protect_last`` messages are never pruned (they are
    likely still relevant), non-``tool`` roles pass through, and the
    input list is never mutated. ``pruned_count`` reports how many
    tool messages were actually rewritten.
    """

    out: list[dict[str, Any]] = []
    last_protected_idx = max(0, len(messages) - protect_last)
    pruned = 0
    for idx, msg in enumerate(messages):
        if msg.get("role") != "tool" or idx >= last_protected_idx:
            out.append(msg)
            continue
        new_content, did_prune = prune_tool_result(
            msg.get("content"),
            max_chars=max_chars,
            head_lines=head_lines,
            tail_lines=tail_lines,
        )
        if did_prune:
            pruned += 1
            new_msg = dict(msg)
            new_msg["content"] = new_content
            out.append(new_msg)
        else:
            out.append(msg)
    return out, pruned


__all__ = [
    "DEFAULT_HEAD_LINES",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_TAIL_LINES",
    "prune_messages",
    "prune_tool_result",
]
