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
"""Transcript module for Xerxes.

Exports:
    - TranscriptEntry
    - TranscriptStore"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TranscriptEntry:
    """Transcript entry.

    Attributes:
        role (str): role.
        content (str): content.
        timestamp (str): timestamp.
        metadata (dict[str, Any]): metadata."""

    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptStore:
    """Transcript store.

    Attributes:
        entries (list[TranscriptEntry]): entries.
        flushed (bool): flushed.
        compaction_count (int): compaction count."""

    entries: list[TranscriptEntry] = field(default_factory=list)
    flushed: bool = False
    compaction_count: int = 0

    def append(self, role: str, content: str, **metadata: Any) -> None:
        """Append.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            role (str): IN: role. OUT: Consumed during execution.
            content (str): IN: content. OUT: Consumed during execution.
            **metadata: IN: Additional keyword arguments. OUT: Passed through to downstream calls."""

        self.entries.append(TranscriptEntry(role=role, content=content, metadata=metadata))
        self.flushed = False

    def append_entry(self, entry: TranscriptEntry) -> None:
        """Append entry.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            entry (TranscriptEntry): IN: entry. OUT: Consumed during execution."""

        self.entries.append(entry)
        self.flushed = False

    def compact(self, keep_last: int = 10) -> int:
        """Compact.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            keep_last (int, optional): IN: keep last. Defaults to 10. OUT: Consumed during execution.
        Returns:
            int: OUT: Result of the operation."""

        if len(self.entries) <= keep_last:
            return 0
        removed = len(self.entries) - keep_last
        self.entries[:] = self.entries[-keep_last:]
        self.compaction_count += 1
        return removed

    def compact_with_summary(self, keep_last: int = 10, summarizer: Any = None) -> int:
        """Compact with summary.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            keep_last (int, optional): IN: keep last. Defaults to 10. OUT: Consumed during execution.
            summarizer (Any, optional): IN: summarizer. Defaults to None. OUT: Consumed during execution.
        Returns:
            int: OUT: Result of the operation."""

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
        """Replay.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            tuple[TranscriptEntry, ...]: OUT: Result of the operation."""

        return tuple(self.entries)

    def to_messages(self) -> list[dict[str, Any]]:
        """To messages.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[dict[str, Any]]: OUT: Result of the operation."""

        messages = []
        for entry in self.entries:
            msg: dict[str, Any] = {"role": entry.role, "content": entry.content}
            msg.update(entry.metadata)
            messages.append(msg)
        return messages

    def flush(self) -> None:
        """Flush.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        self.flushed = True

    def clear(self) -> None:
        """Clear.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        self.entries.clear()
        self.flushed = False

    @property
    def turn_count(self) -> int:
        """Return Turn count.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            int: OUT: Result of the operation."""

        return sum(1 for e in self.entries if e.role == "user")

    @property
    def message_count(self) -> int:
        """Return Message count.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            int: OUT: Result of the operation."""

        return len(self.entries)

    def as_markdown(self) -> str:
        """As markdown.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            str: OUT: Result of the operation."""

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
