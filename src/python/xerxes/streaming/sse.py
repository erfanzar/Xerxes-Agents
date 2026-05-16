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
"""Incremental Server-Sent-Events parser with ``Last-Event-ID`` support.

Used by the streaming-provider clients to recover from transient drops in the
middle of a Responses-API stream. The parser is pure — it consumes raw text
chunks via :meth:`SSEParser.feed`, releases completed records via
:meth:`SSEParser.drain`, and tracks ``last_event_id`` for use as the
``Last-Event-ID`` HTTP header on reconnect. The caller decides whether to
reconnect or abort.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass
class SSEEvent:
    """A single completed SSE record.

    Attributes:
        event: Event name (the value of the ``event:`` field; defaults to
            ``"message"`` per the spec).
        data: Concatenated value of all ``data:`` lines in the record.
        id: Event id (the ``id:`` field), used for ``Last-Event-ID`` resume.
        retry: Suggested reconnect delay in milliseconds, or ``None``.
    """

    event: str = "message"
    data: str = ""
    id: str = ""
    retry: int | None = None


class SSEParser:
    """Incremental Server-Sent-Events parser.

    Feed raw text chunks via :meth:`feed`; pop completed records with
    :meth:`drain`. The parser remembers the last ``id`` it saw on
    :attr:`last_event_id` so the caller can pass it as ``Last-Event-ID`` when
    reconnecting after a drop.
    """

    def __init__(self) -> None:
        """Initialise empty buffers for a fresh stream."""
        self._buffer = ""
        self._current_event: str = "message"
        self._current_data: list[str] = []
        self._current_id: str = ""
        self._current_retry: int | None = None
        self._completed: list[SSEEvent] = []
        self.last_event_id: str = ""

    def feed(self, chunk: str) -> None:
        """Buffer a chunk of raw text and process every complete line."""
        self._buffer += chunk
        while "\n" in self._buffer:
            line, _, rest = self._buffer.partition("\n")
            self._buffer = rest
            self._handle_line(line.rstrip("\r"))

    def _handle_line(self, line: str) -> None:
        """Apply one parsed SSE line; blank lines flush the current record."""
        if not line:
            self._dispatch()
            return
        if line.startswith(":"):
            return  # comment
        if ":" in line:
            field_name, _, value = line.partition(":")
            value = value.lstrip(" ")
        else:
            field_name, value = line, ""
        if field_name == "event":
            self._current_event = value
        elif field_name == "data":
            self._current_data.append(value)
        elif field_name == "id":
            self._current_id = value
        elif field_name == "retry":
            try:
                self._current_retry = int(value)
            except ValueError:
                self._current_retry = None

    def _dispatch(self) -> None:
        """Finalise the in-progress record into an :class:`SSEEvent`."""
        if not self._current_data and not self._current_event != "message":
            return
        ev = SSEEvent(
            event=self._current_event,
            data="\n".join(self._current_data),
            id=self._current_id,
            retry=self._current_retry,
        )
        self._completed.append(ev)
        if ev.id:
            self.last_event_id = ev.id
        self._current_event = "message"
        self._current_data = []
        self._current_id = ""
        self._current_retry = None

    def drain(self) -> list[SSEEvent]:
        """Return and clear all events completed since the last drain."""
        out = self._completed
        self._completed = []
        return out


def parse_sse_stream(chunks: Iterable[str]) -> Iterator[SSEEvent]:
    """Parse a finite chunk stream end-to-end and yield every event.

    Convenience over :class:`SSEParser` for tests and one-shot uses. After
    feeding the chunks, a trailing ``\\n\\n`` is appended to flush any final
    record that lacks a terminating blank line.
    """
    p = SSEParser()
    for chunk in chunks:
        p.feed(chunk)
        for ev in p.drain():
            yield ev
    # Flush any pending event (final lines without trailing blank line).
    p.feed("\n\n")
    for ev in p.drain():
        yield ev


__all__ = ["SSEEvent", "SSEParser", "parse_sse_stream"]
