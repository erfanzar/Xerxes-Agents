"""Tests for calute.core.streamer_buffer module."""

import threading

from calute.core.streamer_buffer import KILL_TAG, StreamerBuffer


class TestStreamerBuffer:
    def test_init(self):
        buf = StreamerBuffer()
        assert buf.closed is False
        assert buf.thread is None
        assert buf.task is None

    def test_put_get(self):
        buf = StreamerBuffer()
        buf.put("hello")
        assert buf.get(timeout=1.0) == "hello"

    def test_get_timeout(self):
        buf = StreamerBuffer()
        result = buf.get(timeout=0.1)
        assert result is None

    def test_close(self):
        buf = StreamerBuffer()
        buf.close()
        assert buf.closed is True

    def test_close_idempotent(self):
        buf = StreamerBuffer()
        buf.close()
        buf.close()
        assert buf.closed is True

    def test_put_after_close(self):
        buf = StreamerBuffer()
        buf.close()
        buf.put("should be dropped")

    def test_stream_basic(self):
        buf = StreamerBuffer()
        buf.put("item1")
        buf.put("item2")
        buf.put(KILL_TAG)

        items = list(buf.stream())
        assert "item1" in items
        assert "item2" in items

    def test_stream_from_thread(self):
        buf = StreamerBuffer()
        results = []

        def producer():
            for i in range(3):
                buf.put(f"msg-{i}")
            buf.close()

        t = threading.Thread(target=producer)
        t.start()

        for item in buf.stream():
            if item is not None:
                results.append(item)

        t.join()
        assert len(results) == 3

    def test_maybe_finish_no_completion(self):
        buf = StreamerBuffer()
        buf.maybe_finish(None)
        assert buf.closed is False

    def test_maybe_finish_with_completion(self):
        buf = StreamerBuffer()
        buf._finish_hit = True
        buf.maybe_finish(None)
        assert buf.closed is True

    def test_maybe_finish_non_none_arg(self):
        buf = StreamerBuffer()
        buf._finish_hit = True
        buf.maybe_finish("something")
        assert buf.closed is False

    def test_maxsize(self):
        buf = StreamerBuffer(maxsize=2)
        buf.put("a")
        buf.put("b")
        assert buf.get(timeout=0.1) == "a"
        assert buf.get(timeout=0.1) == "b"
