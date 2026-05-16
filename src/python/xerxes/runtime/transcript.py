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
"""In-memory transcript store backing every :class:`QueryEngine` session.

:class:`TranscriptEntry` holds one role/content/metadata triple; :class:`TranscriptStore`
keeps the ordered list plus compaction helpers that the query engine uses
when ``compact_after_turns`` is exceeded. The store is also serialised in
``QueryEngine.to_dict`` and converted to provider message dicts via
:meth:`TranscriptStore.to_messages`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TranscriptEntry:
    """One message in the transcript.

    Attributes:
        role: Conversational role (``"user"``, ``"assistant"``, ``"system"``,
            ``"tool"``).
        content: Rendered text content of the message.
        timestamp: ISO-8601 timestamp captured when the entry was constructed.
        metadata: Provider-specific extras (tool call ids, attachments, etc.)
            merged into the message dict by :meth:`TranscriptStore.to_messages`.
    """

    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptStore:
    """Ordered list of :class:`TranscriptEntry` records with compaction helpers.

    Attributes:
        entries: All messages in chronological order.
        flushed: Whether the store has been persisted since its last write.
        compaction_count: Number of times a compaction routine has run.
    """

    entries: list[TranscriptEntry] = field(default_factory=list)
    flushed: bool = False
    compaction_count: int = 0

    def append(self, role: str, content: str, **metadata: Any) -> None:
        """Append a new entry built from ``role``, ``content`` and ``metadata``."""
        self.entries.append(TranscriptEntry(role=role, content=content, metadata=metadata))
        self.flushed = False

    def append_entry(self, entry: TranscriptEntry) -> None:
        """Append an already-constructed :class:`TranscriptEntry`."""
        self.entries.append(entry)
        self.flushed = False

    def compact(self, keep_last: int = 10) -> int:
        """Drop everything except the last ``keep_last`` entries.

        Returns the number of entries removed (zero when no compaction was
        necessary). Increments :attr:`compaction_count` on a non-empty drop.
        """

        if len(self.entries) <= keep_last:
            return 0
        removed = len(self.entries) - keep_last
        self.entries[:] = self.entries[-keep_last:]
        self.compaction_count += 1
        return removed

    def compact_with_summary(self, keep_last: int = 10, summarizer: Any = None) -> int:
        """Compact while preserving a synthesised summary of the dropped span.

        Args:
            keep_last: Number of most-recent entries kept verbatim.
            summarizer: Optional callable ``(entries) -> str`` invoked on the
                slice that would otherwise be discarded. The returned text is
                prepended to the surviving entries as a system message.

        Returns:
            The number of entries that were summarised away.
        """

        if len(self.entries) <= keep_last:
            return 0

        old_entries = self.entries[:-keep_last]
        removed = len(old_entries)

        if summarizer:
            summary = summarizer(old_entries)
            self.entries[:] = [
                TranscriptEntry(role="system", content=f"[Compacted summary]\n{summary}"),
                *self.entries[-keep_last:],
            ]
        else:
            self.entries[:] = self.entries[-keep_last:]

        self.compaction_count += 1
        return removed

    def replay(self) -> tuple[TranscriptEntry, ...]:
        """Return an immutable snapshot of every stored entry, in order."""
        return tuple(self.entries)

    def to_messages(self) -> list[dict[str, Any]]:
        """Render entries as provider-style ``{"role", "content", ...}`` dicts.

        Metadata fields are merged onto each dict so tool-call ids and other
        provider-specific keys round-trip back into the streaming loop.
        """

        messages = []
        for entry in self.entries:
            msg: dict[str, Any] = {"role": entry.role, "content": entry.content}
            msg.update(entry.metadata)
            messages.append(msg)
        return messages

    def flush(self) -> None:
        """Mark the in-memory store as persisted by the caller."""
        self.flushed = True

    def clear(self) -> None:
        """Drop every entry and reset the flushed flag."""
        self.entries.clear()
        self.flushed = False

    @property
    def turn_count(self) -> int:
        """Number of user turns (entries with ``role == "user"``)."""
        return sum(1 for e in self.entries if e.role == "user")

    @property
    def message_count(self) -> int:
        """Total stored entries regardless of role."""
        return len(self.entries)

    def as_markdown(self) -> str:
        """Render the transcript as a truncated Markdown bullet list."""

        lines = ["# Transcript", "", f"Messages: {self.message_count}", ""]
        for entry in self.entries:
            role_tag = f"**{entry.role}**"
            content_preview = entry.content[:200]
            if len(entry.content) > 200:
                content_preview += "..."
            lines.append(f"- {role_tag}: {content_preview}")
        return "\n".join(lines)


__all__ = [
    "TranscriptEntry",
    "TranscriptStore",
]
