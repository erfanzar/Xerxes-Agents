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
"""Audit event collectors.

This module defines several collector implementations for persisting or
buffering audit events: an in-memory collector, a JSONL file sink, and a
composite that forwards to multiple collectors.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any, Protocol, runtime_checkable

from .events import AuditEvent


@runtime_checkable
class AuditCollector(Protocol):
    """Protocol for audit event collectors.

    Implementations must accept events and support flushing.
    """

    def emit(self, event: AuditEvent) -> None:
        """Emit an audit event.

        Args:
            event (AuditEvent): IN: The event to record. OUT: Stored or forwarded.
        """
        ...

    def flush(self) -> None:
        """Flush any buffered events to their final destination."""
        ...


class InMemoryCollector:
    """Thread-safe in-memory collector for audit events."""

    def __init__(self) -> None:
        """Initialize the collector with an empty event list."""
        self._lock = threading.Lock()
        self._events: list[AuditEvent] = []

    def emit(self, event: AuditEvent) -> None:
        """Append an event to the in-memory list.

        Args:
            event (AuditEvent): IN: Event to store. OUT: Appended under lock.
        """
        with self._lock:
            self._events.append(event)

    def flush(self) -> None:
        """No-op for in-memory storage."""

    def get_events(self) -> list[AuditEvent]:
        """Return a copy of all stored events.

        Returns:
            list[AuditEvent]: OUT: Snapshot of the event list.
        """
        with self._lock:
            return list(self._events)

    def get_events_by_type(self, event_type: str) -> list[AuditEvent]:
        """Return events filtered by type.

        Args:
            event_type (str): IN: Event type string to match. OUT: Used as filter.

        Returns:
            list[AuditEvent]: OUT: Matching events.
        """
        with self._lock:
            return [e for e in self._events if e.event_type == event_type]

    def clear(self) -> None:
        """Remove all stored events."""
        with self._lock:
            self._events.clear()

    def __len__(self) -> int:
        """Return the number of stored events.

        Returns:
            int: OUT: Event count.
        """
        with self._lock:
            return len(self._events)


class JSONLSinkCollector:
    """Collector that appends events as JSON Lines to a file or stream."""

    def __init__(self, sink: str | Path | IO[str]) -> None:
        """Initialize the JSONL collector.

        Args:
            sink (str | Path | IO[str]): IN: File path or open text stream. OUT:
                If a path, opened in append mode and owned by this collector.
        """
        self._lock = threading.Lock()
        self._owns_stream = False

        if isinstance(sink, str | Path):
            self._stream: IO[str] = open(sink, "a", encoding="utf-8")
            self._owns_stream = True
        else:
            self._stream = sink

    def emit(self, event: AuditEvent) -> None:
        """Serialize an event to JSON and append a line.

        Args:
            event (AuditEvent): IN: Event to persist. OUT: Serialized and written.
        """
        line = json.dumps(event.to_dict(), default=str) + "\n"
        with self._lock:
            self._stream.write(line)

    def flush(self) -> None:
        """Flush the underlying stream."""
        with self._lock:
            self._stream.flush()

    def close(self) -> None:
        """Flush and close the underlying stream if owned."""
        self.flush()
        if self._owns_stream:
            self._stream.close()


class CompositeCollector:
    """Collector that forwards events to multiple child collectors."""

    def __init__(
        self,
        collectors: Sequence[AuditCollector | InMemoryCollector | JSONLSinkCollector | CompositeCollector] | None = None,
    ) -> None:
        """Initialize the composite collector.

        Args:
            collectors (Sequence | None): IN: Child collectors. OUT: Stored for
                event forwarding.
        """
        self._collectors: list[Any] = list(collectors or [])

    def add(self, collector: AuditCollector | InMemoryCollector | JSONLSinkCollector) -> None:
        """Add a child collector.

        Args:
            collector (AuditCollector | InMemoryCollector | JSONLSinkCollector):
                IN: Collector to append. OUT: Added to the internal list.
        """
        self._collectors.append(collector)

    def emit(self, event: AuditEvent) -> None:
        """Forward an event to all child collectors.

        Args:
            event (AuditEvent): IN: Event to broadcast. OUT: Passed to each child.
        """
        for collector in self._collectors:
            collector.emit(event)

    def flush(self) -> None:
        """Flush all child collectors."""
        for collector in self._collectors:
            collector.flush()
