# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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

"""Audit event collectors for buffering, persisting, and fanning-out events.

Collectors implement a simple two-method protocol defined by
:class:`AuditCollector`:

    * :meth:`~AuditCollector.emit` -- accept a single audit event.
    * :meth:`~AuditCollector.flush` -- flush any buffered state to the
      backing store.

Three concrete implementations ship with this module:

    * :class:`InMemoryCollector` -- thread-safe in-memory list with
      filtering and snapshot capabilities.
    * :class:`JSONLSinkCollector` -- streams each event as a single
      newline-delimited JSON line to a file or IO stream.
    * :class:`CompositeCollector` -- fans out every event to an
      arbitrary number of child collectors.

Example:
    Wiring an in-memory collector and a JSONL file sink together::

        from calute.audit.collector import (
            CompositeCollector,
            InMemoryCollector,
            JSONLSinkCollector,
        )

        mem = InMemoryCollector()
        jsonl = JSONLSinkCollector("/tmp/audit.jsonl")
        composite = CompositeCollector([mem, jsonl])
        composite.emit(some_event)
        composite.flush()
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
    """Protocol that all audit collectors must satisfy.

    Any object that implements :meth:`emit` and :meth:`flush` is
    considered a valid audit collector. The protocol is decorated with
    :func:`~typing.runtime_checkable` so that ``isinstance`` checks
    work at runtime.

    Example:
        Implementing a custom collector::

            class MyCollector:
                def emit(self, event: AuditEvent) -> None:
                    print(event.to_json())

                def flush(self) -> None:
                    pass

            assert isinstance(MyCollector(), AuditCollector)
    """

    def emit(self, event: AuditEvent) -> None:
        """Accept a single audit event and process or store it.

        Args:
            event: The audit event to collect.
        """
        ...

    def flush(self) -> None:
        """Flush any buffered state to the collector's backing store.

        Implementations that write to IO streams or network sockets
        should ensure all pending data is written. In-memory collectors
        may treat this as a no-op.
        """
        ...


class InMemoryCollector:
    """Thread-safe in-memory audit event buffer.

    Events are stored in insertion order and can be retrieved, filtered,
    or cleared at any time. All public methods acquire an internal
    :class:`threading.Lock` to guarantee safe concurrent access.

    Attributes:
        _lock: Internal threading lock for thread-safe access.
        _events: Internal list storing collected audit events.

    Example:
        ::

            collector = InMemoryCollector()
            collector.emit(TurnStartEvent(agent_id="a1", prompt_preview="Hi"))
            assert len(collector) == 1
            events = collector.get_events()
            collector.clear()
            assert len(collector) == 0
    """

    def __init__(self) -> None:
        """Initialize an empty in-memory collector with a threading lock."""
        self._lock = threading.Lock()
        self._events: list[AuditEvent] = []

    def emit(self, event: AuditEvent) -> None:
        """Append an audit event to the in-memory buffer.

        The event is appended under the internal lock to ensure
        thread-safe writes from concurrent producers.

        Args:
            event: The audit event to store.
        """
        with self._lock:
            self._events.append(event)

    def flush(self) -> None:
        """No-op for in-memory storage.

        Provided for protocol compatibility with :class:`AuditCollector`.
        """

    def get_events(self) -> list[AuditEvent]:
        """Return a shallow-copy snapshot of all collected events.

        The returned list is a copy, so callers may mutate it without
        affecting the internal buffer.

        Returns:
            list[AuditEvent]: A new list containing all events in
                insertion order.
        """
        with self._lock:
            return list(self._events)

    def get_events_by_type(self, event_type: str) -> list[AuditEvent]:
        """Return events whose ``event_type`` matches the given string.

        Args:
            event_type: The event type tag to filter on
                (e.g. ``"turn_start"``, ``"tool_call_attempt"``).

        Returns:
            list[AuditEvent]: A filtered list of matching events in
                insertion order.
        """
        with self._lock:
            return [e for e in self._events if e.event_type == event_type]

    def clear(self) -> None:
        """Discard all collected events from the internal buffer.

        After this call, :meth:`get_events` will return an empty list
        and ``len(self)`` will return ``0``.
        """
        with self._lock:
            self._events.clear()

    def __len__(self) -> int:
        """Return the number of events currently in the buffer.

        Returns:
            int: The event count.
        """
        with self._lock:
            return len(self._events)


class JSONLSinkCollector:
    """Writes each event as a single JSON line to a file or IO stream.

    Each call to :meth:`emit` serializes the event via
    :meth:`~calute.audit.events.AuditEvent.to_dict` and writes it as a
    single newline-terminated JSON object. All writes are protected by
    an internal :class:`threading.Lock`.

    When a filesystem path is provided, the file is opened in **append**
    mode so that existing audit logs are preserved across restarts.

    Attributes:
        _lock: Internal threading lock for thread-safe writes.
        _owns_stream: Whether this collector owns (and should close) the
            underlying IO stream.
        _stream: The writable IO stream that receives JSON lines.

    Args:
        sink: Either a filesystem path (``str`` or ``pathlib.Path``) or
            an already-open writable text IO stream. When a path is
            given the file is opened in append mode with UTF-8 encoding.

    Example:
        Writing to a file::

            collector = JSONLSinkCollector("/tmp/audit.jsonl")
            collector.emit(some_event)
            collector.flush()
            collector.close()

        Writing to an in-memory stream::

            import io
            buf = io.StringIO()
            collector = JSONLSinkCollector(buf)
            collector.emit(some_event)
            print(buf.getvalue())
    """

    def __init__(self, sink: str | Path | IO[str]) -> None:
        """Initialize the JSONL sink collector.

        Args:
            sink: A filesystem path (``str`` or ``pathlib.Path``) to
                open in append mode, or an already-open writable text IO
                stream.
        """
        self._lock = threading.Lock()
        self._owns_stream = False

        if isinstance(sink, str | Path):
            self._stream: IO[str] = open(sink, "a", encoding="utf-8")
            self._owns_stream = True
        else:
            self._stream = sink

    def emit(self, event: AuditEvent) -> None:
        """Serialize the event to a JSON line and write it to the stream.

        Non-serializable values in the event dict are coerced to strings
        via the ``default=str`` JSON encoder fallback.

        Args:
            event: The audit event to serialize and write.
        """
        line = json.dumps(event.to_dict(), default=str) + "\n"
        with self._lock:
            self._stream.write(line)

    def flush(self) -> None:
        """Flush the underlying IO stream.

        Ensures that all buffered data has been written to the stream's
        backing store (e.g. the operating system's file buffer).
        """
        with self._lock:
            self._stream.flush()

    def close(self) -> None:
        """Flush pending writes and close the stream if owned.

        If this collector opened the stream itself (i.e. a path was
        passed to the constructor), the stream is closed after flushing.
        Externally-supplied streams are flushed but left open so that
        the caller retains control of the stream lifecycle.
        """
        self.flush()
        if self._owns_stream:
            self._stream.close()


class CompositeCollector:
    """Fans out every emitted event to multiple child collectors.

    This collector acts as a multiplexer: each call to :meth:`emit`
    forwards the event to every registered child, and :meth:`flush`
    flushes all children in sequence. New children can be added at any
    time via :meth:`add`.

    Attributes:
        _collectors: Internal list of child collectors that receive
            forwarded events.

    Args:
        collectors: An optional initial sequence of child collectors.
            When ``None`` (the default), the composite starts empty.

    Example:
        ::

            mem = InMemoryCollector()
            jsonl = JSONLSinkCollector("/tmp/audit.jsonl")
            composite = CompositeCollector([mem, jsonl])
            composite.emit(some_event)
            assert len(mem) == 1
    """

    def __init__(
        self,
        collectors: Sequence[AuditCollector | InMemoryCollector | JSONLSinkCollector | CompositeCollector] | None = None,
    ) -> None:
        """Initialize the composite collector with optional children.

        Args:
            collectors: An optional sequence of child collectors to
                register immediately. Defaults to an empty list when
                ``None``.
        """
        self._collectors: list[Any] = list(collectors or [])

    def add(self, collector: AuditCollector | InMemoryCollector | JSONLSinkCollector) -> None:
        """Append a collector to the fan-out list.

        Args:
            collector: The child collector to register. It must
                implement the :class:`AuditCollector` protocol (i.e.
                provide ``emit`` and ``flush`` methods).
        """
        self._collectors.append(collector)

    def emit(self, event: AuditEvent) -> None:
        """Forward the event to every registered child collector.

        Children are called in registration order. If a child raises an
        exception, subsequent children will **not** receive the event.

        Args:
            event: The audit event to broadcast.
        """
        for collector in self._collectors:
            collector.emit(event)

    def flush(self) -> None:
        """Flush every registered child collector in registration order.

        Each child's :meth:`flush` is called sequentially. If a child
        raises an exception, subsequent children will **not** be flushed.
        """
        for collector in self._collectors:
            collector.flush()
