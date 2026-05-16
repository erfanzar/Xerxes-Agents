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
"""Thread-safe buffer that bridges async stream producers to sync consumers.

LLM providers feed chunks from an async producer (HTTP stream task), while
the agent loop iterates them synchronously. :class:`StreamerBuffer` wraps a
``queue.Queue`` so the producer ``put``s items, the consumer pulls them via
:meth:`StreamerBuffer.stream`, and either side can request a graceful
shutdown by sending the sentinel :data:`KILL_TAG`. The buffer is fully
thread-safe; closing it idempotently signals every consumer.
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
    """Thread-safe queue carrying streaming chunks plus producer/consumer bookkeeping.

    The optional ``thread``/``task``/``result_holder``/``exception_holder``
    fields are populated by helpers that adopt the buffer's lifetime to a
    background producer (so the consumer can wait on the producer's result
    via :attr:`get_result` / :attr:`aget_result` after :meth:`stream` exits).
    """

    def __init__(self, maxsize: int = 0):
        """Build the queue (``maxsize=0`` means unbounded)."""
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
        """Enqueue ``item`` (``None`` is treated as a heartbeat by consumers); drops if closed."""
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
        """Pop the next item, returning ``None`` on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stream(self) -> Generator[StreamingResponseType, None, None]:
        """Iterate items as they arrive, exiting cleanly when :data:`KILL_TAG` is dequeued."""
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
        """Idempotently close the buffer; enqueues :data:`KILL_TAG` so consumers can exit."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._queue.put(tp.cast(StreamingResponseType, KILL_TAG))

    @property
    def closed(self) -> bool:
        """``True`` once :meth:`close` has been called."""
        return self._closed

    def maybe_finish(self, arg: tp.Any) -> None:
        """Close the buffer when ``arg`` is ``None`` after a :class:`Completion` was seen."""
        if arg is None and self._finish_hit:
            self.close()


__all__ = ("StreamerBuffer",)
