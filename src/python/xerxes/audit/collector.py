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
"""Sinks for :class:`AuditEvent` records.

Three concrete implementations are provided plus the
:class:`AuditCollector` protocol they satisfy:

* :class:`InMemoryCollector` — thread-safe list, useful for tests and
  in-process consumers (e.g. the session viewer).
* :class:`JSONLSinkCollector` — appends one JSON object per line to a
  path or stream; the canonical on-disk format.
* :class:`CompositeCollector` — fans events out to several collectors.
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
    """Minimal protocol every audit sink must satisfy."""

    def emit(self, event: AuditEvent) -> None:
        """Record ``event``."""
        ...

    def flush(self) -> None:
        """Flush any buffered events to their final destination."""
        ...


class InMemoryCollector:
    """Thread-safe in-memory ring of events, primarily for tests."""

    def __init__(self) -> None:
        """Start with an empty event list and an internal lock."""
        self._lock = threading.Lock()
        self._events: list[AuditEvent] = []

    def emit(self, event: AuditEvent) -> None:
        """Append ``event`` to the in-memory list."""
        with self._lock:
            self._events.append(event)

    def flush(self) -> None:
        """No buffered state; provided for protocol compatibility."""

    def get_events(self) -> list[AuditEvent]:
        """Return a shallow copy of every recorded event."""
        with self._lock:
            return list(self._events)

    def get_events_by_type(self, event_type: str) -> list[AuditEvent]:
        """Return events whose discriminator equals ``event_type``."""
        with self._lock:
            return [e for e in self._events if e.event_type == event_type]

    def clear(self) -> None:
        """Drop every recorded event."""
        with self._lock:
            self._events.clear()

    def __len__(self) -> int:
        """Return the number of recorded events."""
        with self._lock:
            return len(self._events)


class JSONLSinkCollector:
    """Appends events as JSON Lines to a file or open text stream."""

    def __init__(self, sink: str | Path | IO[str]) -> None:
        """Open ``sink`` for append, or attach to a pre-opened stream.

        When given a path, the collector owns the file and will close
        it on :meth:`close`. Pre-opened streams are left for the caller.
        """
        self._lock = threading.Lock()
        self._owns_stream = False

        if isinstance(sink, str | Path):
            self._stream: IO[str] = open(sink, "a", encoding="utf-8")
            self._owns_stream = True
        else:
            self._stream = sink

    def emit(self, event: AuditEvent) -> None:
        """Serialise ``event`` to one JSON line and write it under lock."""
        line = json.dumps(event.to_dict(), default=str) + "\n"
        with self._lock:
            self._stream.write(line)

    def flush(self) -> None:
        """Flush buffered writes to the underlying stream."""
        with self._lock:
            self._stream.flush()

    def close(self) -> None:
        """Flush and, if this collector owns the stream, close it."""
        self.flush()
        if self._owns_stream:
            self._stream.close()


class CompositeCollector:
    """Fan-out collector that forwards every event to its children."""

    def __init__(
        self,
        collectors: Sequence[AuditCollector | InMemoryCollector | JSONLSinkCollector | CompositeCollector] | None = None,
    ) -> None:
        """Wrap ``collectors``; new children may be added via :meth:`add`."""
        self._collectors: list[Any] = list(collectors or [])

    def add(self, collector: AuditCollector | InMemoryCollector | JSONLSinkCollector) -> None:
        """Append ``collector`` to the fan-out list."""
        self._collectors.append(collector)

    def emit(self, event: AuditEvent) -> None:
        """Forward ``event`` to each child in registration order."""
        for collector in self._collectors:
            collector.emit(event)

    def flush(self) -> None:
        """Flush every child collector."""
        for collector in self._collectors:
            collector.flush()
