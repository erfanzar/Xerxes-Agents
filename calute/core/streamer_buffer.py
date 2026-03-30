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


"""Streaming buffer utilities for response handling.

This module provides buffering infrastructure for streaming responses in Calute,
including:
- Thread-safe queue-based buffering for streaming data
- Support for both synchronous and asynchronous consumption
- Graceful shutdown and cleanup mechanisms
- Debug logging for troubleshooting streaming issues

The StreamerBuffer class enables efficient streaming of responses from
background threads or async tasks to the main consumer, with proper
lifecycle management and error handling.
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
    """Thread-safe buffer for streaming responses with put/get interface.

    Provides a queue-based buffering mechanism for streaming responses from
    background threads or async tasks. Supports both blocking and non-blocking
    access patterns, and handles graceful shutdown via a kill signal.

    Attributes:
        thread: Optional thread running the streaming producer.
        task: Optional asyncio task for async streaming operations.
        result_holder: Optional list to store the final result.
        exception_holder: Optional list to store exceptions during streaming.
        get_result: Optional callable to retrieve the final result synchronously.
        aget_result: Optional callable to retrieve the final result asynchronously.
    """

    def __init__(self, maxsize: int = 0):
        """Initialize the StreamerBuffer.

        Creates a new thread-safe streaming buffer backed by a
        :class:`queue.Queue`. The buffer starts in the open state and
        can be permanently closed via :meth:`close`.

        Args:
            maxsize: Maximum number of items the internal queue can hold.
                A value of 0 (the default) means the queue is unbounded.
        """
        self._queue: queue.Queue[StreamingResponseType | None] = queue.Queue(maxsize=maxsize)
        self._closed = False
        self._lock = threading.Lock()
        self._finish_hit = False
        self.thread: threading.Thread | None = None
        self.task: asyncio.Task | None = None  # type: ignore
        self.result_holder: list[tp.Any | None] | None = None
        self.exception_holder: list[Exception | None] | None = None
        self.get_result: tp.Callable[[float | None], tp.Any] | None = None
        self.aget_result: tp.Callable[[], tp.Awaitable[tp.Any]] | None = None

    def put(self, item: StreamingResponseType | None) -> None:
        """Put an item into the buffer.

        Adds a streaming response item to the internal queue. If the buffer
        is closed, the item will be dropped (with a warning if debug mode
        is enabled).

        Args:
            item: The streaming response to buffer (None signals end of current stream).

        Returns:
            None
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
        """Get an item from the buffer.

        Retrieves and removes an item from the internal queue. Blocks until
        an item is available or the timeout expires.

        Args:
            timeout: Timeout in seconds (None for blocking indefinitely).

        Returns:
            The streaming response item, or None if the timeout expired
            without an item becoming available.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stream(self) -> Generator[StreamingResponseType, None, None]:
        """Generator that yields all items from buffer until terminated.

        Continuously retrieves items from the buffer and yields them until
        a kill signal (KILL_TAG) is received. Automatically tracks when
        a Completion item is encountered to support graceful shutdown.

        Yields:
            StreamingResponseType: Streaming response items from the buffer.
        """
        while True:
            try:
                item = self.get(timeout=1.0)
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
        """Permanently close the buffer.

        Marks the buffer as closed and sends a kill signal to terminate
        any active stream consumers. This operation is thread-safe and
        idempotent - calling close() multiple times has no additional effect.

        Returns:
            None
        """
        with self._lock:
            if not self._closed:
                self._closed = True
                self._queue.put(KILL_TAG)

    @property
    def closed(self) -> bool:
        """Check if buffer is closed.

        Returns:
            True if the buffer has been permanently closed, False otherwise.
        """
        return self._closed

    def maybe_finish(self, arg: tp.Any) -> None:
        """Conditionally close the buffer based on completion state.

        Closes the buffer if the provided argument is None and a Completion
        item has been previously encountered during streaming. This enables
        automatic cleanup when the stream has naturally completed.

        Args:
            arg: The argument to check. If None and a completion was seen,
                the buffer will be closed.

        Returns:
            None
        """
        if arg is None and self._finish_hit:
            self.close()


__all__ = ("StreamerBuffer",)
