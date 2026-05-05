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
"""Thread-safe buffer for streaming LLM responses.

``StreamerBuffer`` bridges asynchronous producers and synchronous consumers
via a ``queue.Queue``, yielding items through a generator and supporting
graceful shutdown via ``KILL_TAG``.
"""

import os
import queue
import threading
import typing as tp
from collections.abc import Generator

from ..types import StreamingResponseType
from ..types.function_execution_types import Completion

DEBUG_STREAMING = os.environ.get("DEBUG_STREAMING", "").lower() in ["1", "true", "yes"]

if tp.TYPE_CHECKING:
    import asyncio

KILL_TAG = "/<[KILL-LOOP]>/"


class StreamerBuffer:
    """Thread-safe queue wrapper for streaming response chunks."""

    def __init__(self, maxsize: int = 0):
        """Initialize the buffer.

        Args:
            maxsize (int): IN: maximum queue size (0 = unlimited).
                Defaults to 0.
        """
        self._queue: queue.Queue[StreamingResponseType | None] = queue.Queue(maxsize=maxsize)
        self._closed = False
        self._lock = threading.Lock()
        self._finish_hit = False
        self.thread: threading.Thread | None = None
        self.task: asyncio.Task | None = None
        self.result_holder: list[tp.Any | None] | None = None
        self.exception_holder: list[Exception | None] | None = None
        self.get_result: tp.Callable[[float | None], tp.Any] | None = None
        self.aget_result: tp.Callable[[], tp.Awaitable[tp.Any]] | None = None

    def put(self, item: StreamingResponseType | None) -> None:
        """Enqueue an item (or ``None`` as a heartbeat).

        Args:
            item (StreamingResponseType | None): IN: chunk to enqueue.
        """
        if DEBUG_STREAMING:
            import sys

            if item is None:
                print("[StreamerBuffer] Received None signal", file=sys.stderr)

        if not self._closed:
            self._queue.put(item)

        elif DEBUG_STREAMING:
            import sys

            print("[StreamerBuffer] WARNING: Buffer closed, dropping item", file=sys.stderr)

    def get(self, timeout: float | None = None) -> StreamingResponseType | None:
        """Dequeue an item, optionally blocking up to ``timeout``.

        Args:
            timeout (float | None): IN: seconds to wait. ``None`` blocks
                indefinitely.

        Returns:
            StreamingResponseType | None: OUT: dequeued item, or ``None`` on
            timeout.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stream(self) -> Generator[StreamingResponseType, None, None]:
        """Yield items from the buffer until ``KILL_TAG`` is received.

        Yields:
            StreamingResponseType: OUT: individual response chunks.
        """
        while True:
            try:
                item = self.get(timeout=1.0)
                if item is None:
                    continue
                if item is KILL_TAG:
                    if DEBUG_STREAMING:
                        import sys

                        print("[StreamerBuffer.stream] Received KILL_TAG, ending stream", file=sys.stderr)
                    break
                if isinstance(item, Completion):
                    self._finish_hit = True
                yield item
            except queue.Empty:
                continue

    def close(self) -> None:
        """Close the buffer and signal consumers to stop."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._queue.put(tp.cast(StreamingResponseType, KILL_TAG))

    @property
    def closed(self) -> bool:
        """Whether the buffer has been closed.

        Returns:
            bool: OUT: ``True`` if closed.
        """
        return self._closed

    def maybe_finish(self, arg: tp.Any) -> None:
        """Close the buffer if the argument is ``None`` and a finish signal was seen.

        Args:
            arg (Any): IN: arbitrary value (typically a function result).
        """
        if arg is None and self._finish_hit:
            self.close()


__all__ = ("StreamerBuffer",)
