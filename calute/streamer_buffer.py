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


import os
import queue
import threading
import typing as tp
from collections.abc import Generator

from calute.types.function_execution_types import Completion

from .types import StreamingResponseType

DEBUG_STREAMING = os.environ.get("DEBUG_STREAMING", "").lower() in ["1", "true", "yes"]

if tp.TYPE_CHECKING:
    import asyncio

KILL_TAG = "/<[KILL-LOOP]>/"


class StreamerBuffer:
    """Simple buffer for streaming responses with put/get interface."""

    def __init__(self, maxsize: int = 0):
        """
        Initialize the StreamerBuffer.

        Args:
            maxsize: Maximum buffer size (0 for unlimited)
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
        """
        Put an item into the buffer.

        Args:
            item: The streaming response to buffer (None signals end of current stream)
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
        """
        Get an item from the buffer.

        Args:
            timeout: Timeout in seconds (None for blocking)

        Returns:
            The streaming response or None if stream ended
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stream(self) -> Generator[StreamingResponseType, None, None]:
        """
        Generator that yields all items from buffer until None.

        Yields:
            Streaming responses from the buffer
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
        """Permanently close the buffer."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._queue.put(KILL_TAG)

    @property
    def closed(self) -> bool:
        """Check if buffer is closed."""
        return self._closed

    def maybe_finish(self, arg):
        if arg is None and self._finish_hit:
            self.close()


__all__ = ("StreamerBuffer",)
