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
"""Strip <think>/<reasoning> tags from streamed text without losing data.

The streaming/loop already separates a TextChunk vs ThinkingChunk for
``<think>`` tags, but some models leak
``<reasoning>`` / ``<REASONING_SCRATCHPAD>`` blocks into the visible
text stream. ``ReasoningFilter`` is an incremental stripper that the
TUI can wrap around the inbound text stream so the user sees clean
output. Stripped content is captured for the thinking pane.

Exports:
    - ReasoningFilter
    - SUPPRESSED_OPEN_TAGS
    - SUPPRESSED_CLOSE_TAGS"""

from __future__ import annotations

from dataclasses import dataclass, field

SUPPRESSED_OPEN_TAGS: tuple[str, ...] = (
    "<think>",
    "<thinking>",
    "<reasoning>",
    "<reasoning_scratchpad>",
    "<scratchpad>",
)

SUPPRESSED_CLOSE_TAGS: tuple[str, ...] = (
    "</think>",
    "</thinking>",
    "</reasoning>",
    "</reasoning_scratchpad>",
    "</scratchpad>",
)


@dataclass
class _Output:
    """Per-feed accumulator holding the visible and thinking partitions."""

    visible: str = ""
    thinking: str = ""


@dataclass
class ReasoningFilter:
    """Incremental tag-stripper. Feed chunks, get back ``(visible, thinking)``.

    The filter tolerates a tag that arrives split across chunks: it
    buffers a small tail (≤ 32 chars) and re-checks on the next feed.

    Attributes:
        case_insensitive: match tags case-insensitively (default True).
        thinking_log: accumulator of every stripped chunk (visible to
            callers who want to show a reasoning pane)."""

    case_insensitive: bool = True
    _buffer: str = ""
    _in_block: bool = False
    thinking_log: str = field(default="")

    def feed(self, chunk: str) -> tuple[str, str]:
        """Push ``chunk``; return ``(visible_part, thinking_part)``."""
        self._buffer += chunk
        out = _Output()
        while self._buffer:
            if self._in_block:
                # Look for the matching close tag.
                close_idx, close_tag = self._find_any(self._buffer, SUPPRESSED_CLOSE_TAGS)
                if close_idx == -1:
                    # Keep buffering; entire chunk is thinking text.
                    if len(self._buffer) > 32:
                        out.thinking += self._buffer[:-32]
                        self._buffer = self._buffer[-32:]
                    break
                out.thinking += self._buffer[:close_idx]
                self._buffer = self._buffer[close_idx + len(close_tag) :]
                self._in_block = False
                continue
            # Not in a block: look for an open tag.
            open_idx, open_tag = self._find_any(self._buffer, SUPPRESSED_OPEN_TAGS)
            if open_idx == -1:
                # No open tag in the buffer. Emit most of it as visible,
                # keep a short tail in case a tag straddles the boundary.
                if len(self._buffer) > 32:
                    out.visible += self._buffer[:-32]
                    self._buffer = self._buffer[-32:]
                break
            out.visible += self._buffer[:open_idx]
            self._buffer = self._buffer[open_idx + len(open_tag) :]
            self._in_block = True
            continue
        self.thinking_log += out.thinking
        return out.visible, out.thinking

    def flush(self) -> tuple[str, str]:
        """Return any buffered visible/thinking text and reset.

        Call this when the stream completes — otherwise small tails
        get left behind."""
        visible = ""
        thinking = ""
        if self._in_block:
            thinking = self._buffer
        else:
            visible = self._buffer
        self.thinking_log += thinking
        self._buffer = ""
        self._in_block = False
        return visible, thinking

    def _find_any(self, text: str, tags: tuple[str, ...]) -> tuple[int, str]:
        """Return ``(index, tag)`` for the earliest match of any ``tags`` in ``text``.

        Returns ``(-1, "")`` when no tag is present. Respects
        :attr:`case_insensitive` to allow mixed-case model output."""
        haystack = text.lower() if self.case_insensitive else text
        earliest = -1
        earliest_tag = ""
        for tag in tags:
            needle = tag.lower() if self.case_insensitive else tag
            idx = haystack.find(needle)
            if idx != -1 and (earliest == -1 or idx < earliest):
                earliest = idx
                earliest_tag = tag
        return earliest, earliest_tag


__all__ = ["SUPPRESSED_CLOSE_TAGS", "SUPPRESSED_OPEN_TAGS", "ReasoningFilter"]
