# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""Transcript store for conversation message tracking.

Provides a mutable message log with compaction support for managing
long conversations within context limits. Messages can be appended,
compacted (keeping only recent entries), replayed, and serialized.

Inspired by the claw-code ``TranscriptStore`` pattern.

Usage::

    from xerxes.runtime.transcript import TranscriptStore

    store = TranscriptStore()
    store.append("user", "Hello")
    store.append("assistant", "Hi there!")
    store.compact(keep_last=10)
    messages = store.replay()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TranscriptEntry:
    """A single transcript entry.

    Attributes:
        role: Message role (user, assistant, tool, system).
        content: Message content.
        timestamp: When the entry was created.
        metadata: Optional metadata (tool_call_id, name, etc.).
    """

    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptStore:
    """Mutable message transcript with compaction support.

    Attributes:
        entries: List of transcript entries.
        flushed: Whether the transcript has been flushed/persisted.
        compaction_count: Number of times compact() has been called.
    """

    entries: list[TranscriptEntry] = field(default_factory=list)
    flushed: bool = False
    compaction_count: int = 0

    def append(self, role: str, content: str, **metadata: Any) -> None:
        """Add a message to the transcript."""
        self.entries.append(TranscriptEntry(role=role, content=content, metadata=metadata))
        self.flushed = False

    def append_entry(self, entry: TranscriptEntry) -> None:
        """Add a pre-built entry to the transcript."""
        self.entries.append(entry)
        self.flushed = False

    def compact(self, keep_last: int = 10) -> int:
        """Compact the transcript, keeping only the last N entries.

        Args:
            keep_last: Number of recent entries to keep.

        Returns:
            Number of entries removed.
        """
        if len(self.entries) <= keep_last:
            return 0
        removed = len(self.entries) - keep_last
        self.entries[:] = self.entries[-keep_last:]
        self.compaction_count += 1
        return removed

    def compact_with_summary(self, keep_last: int = 10, summarizer: Any = None) -> int:
        """Compact with an optional summary of removed entries.

        If a summarizer is provided, the removed entries are summarized
        and prepended as a system message.

        Args:
            keep_last: Number of recent entries to keep.
            summarizer: Optional callable(entries) -> str.

        Returns:
            Number of entries removed.
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
        """Return all entries as an immutable tuple."""
        return tuple(self.entries)

    def to_messages(self) -> list[dict[str, Any]]:
        """Convert to neutral message format (list of dicts)."""
        messages = []
        for entry in self.entries:
            msg: dict[str, Any] = {"role": entry.role, "content": entry.content}
            msg.update(entry.metadata)
            messages.append(msg)
        return messages

    def flush(self) -> None:
        """Mark the transcript as flushed/persisted."""
        self.flushed = True

    def clear(self) -> None:
        """Remove all entries."""
        self.entries.clear()
        self.flushed = False

    @property
    def turn_count(self) -> int:
        """Number of user messages in the transcript."""
        return sum(1 for e in self.entries if e.role == "user")

    @property
    def message_count(self) -> int:
        """Total number of entries."""
        return len(self.entries)

    def as_markdown(self) -> str:
        """Render the transcript as markdown."""
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
