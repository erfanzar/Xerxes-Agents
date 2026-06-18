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
"""Tests for :mod:`xerxes.runtime.tool_cache`."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from xerxes.runtime.tool_cache import ToolOutputCache


@pytest.fixture
def test_file(tmp_path: Path) -> Path:
    f = tmp_path / "data.py"
    f.write_text("x = 1\n")
    return f


class TestCacheHitMiss:
    """Basic cache hit/miss behavior."""

    def test_first_call_misses_second_hits(self, test_file: Path):
        cache = ToolOutputCache()
        calls: list[str] = []

        def executor(name: str, inp: dict) -> str:
            calls.append(name)
            return "content"

        cached = cache.wrap(executor)
        r1 = cached("ReadFile", {"file_path": str(test_file)})
        r2 = cached("ReadFile", {"file_path": str(test_file)})
        assert r1 == "content"
        assert r2 == "content"
        assert len(calls) == 1
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 1

    def test_different_args_cache_separately(self, test_file: Path):
        cache = ToolOutputCache()
        calls: list[tuple[str, dict]] = []

        def executor(name: str, inp: dict) -> str:
            calls.append((name, inp))
            return f"result for {inp.get('file_path')}"

        cached = cache.wrap(executor)
        cached("ReadFile", {"file_path": str(test_file)})
        cached("ReadFile", {"file_path": str(test_file) + ".bak"})
        assert len(calls) == 2

    def test_non_cacheable_tool_passes_through(self):
        cache = ToolOutputCache()
        calls: list[str] = []

        def executor(name: str, inp: dict) -> str:
            calls.append(name)
            return "written"

        cached = cache.wrap(executor)
        cached("FileEditTool", {"file_path": "x.py", "old_string": "a", "new_string": "b"})
        cached("FileEditTool", {"file_path": "x.py", "old_string": "a", "new_string": "b"})
        assert len(calls) == 2
        assert cache.size == 0


class TestMtimeInvalidation:
    """Cache invalidates when file modification time changes."""

    def test_file_change_invalidates(self, test_file: Path):
        cache = ToolOutputCache()
        calls: list[str] = []

        def executor(name: str, inp: dict) -> str:
            calls.append(name)
            return test_file.read_text()

        cached = cache.wrap(executor)
        cached("ReadFile", {"file_path": str(test_file)})
        assert len(calls) == 1

        # Modify file (update mtime)
        time.sleep(0.01)
        test_file.write_text("x = 2\n")
        os.utime(str(test_file), ns=(time.time_ns(), time.time_ns()))

        cached("ReadFile", {"file_path": str(test_file)})
        assert len(calls) == 2


class TestTtlExpiry:
    """TTL-based cache expiry."""

    def test_expired_entries_are_not_returned(self, test_file: Path):
        cache = ToolOutputCache(ttl_seconds=0.01)
        calls: list[str] = []

        def executor(name: str, inp: dict) -> str:
            calls.append(name)
            return "content"

        cached = cache.wrap(executor)
        cached("ReadFile", {"file_path": str(test_file)})
        assert len(calls) == 1

        time.sleep(0.02)
        cached("ReadFile", {"file_path": str(test_file)})
        assert len(calls) == 2


class TestEviction:
    """LRU eviction when max_entries is exceeded."""

    def test_lru_eviction(self):
        cache = ToolOutputCache(max_entries=3)
        for i in range(5):
            cache.put("ReadFile", {"file_path": f"file_{i}.py"}, f"content_{i}")
        assert cache.size == 3

    def test_lru_order_updates_on_access(self):
        cache = ToolOutputCache(max_entries=2)
        cache.put("ReadFile", {"file_path": "a.py"}, "content_a")
        cache.put("ReadFile", {"file_path": "b.py"}, "content_b")
        # Access a.py — it should now be most-recently-used
        cache.get("ReadFile", {"file_path": "a.py"})
        # Add c.py — b.py should be evicted (LRU), not a.py
        cache.put("ReadFile", {"file_path": "c.py"}, "content_c")
        assert cache.get("ReadFile", {"file_path": "a.py"}) == "content_a"
        assert cache.get("ReadFile", {"file_path": "b.py"}) is None


class TestInvalidate:
    """Manual cache invalidation."""

    def test_invalidate_all(self):
        cache = ToolOutputCache()
        cache.put("ReadFile", {"file_path": "a.py"}, "a")
        cache.put("GrepTool", {"pattern": "x"}, "grep result")
        cache.invalidate()
        assert cache.size == 0

    def test_invalidate_by_tool_name(self):
        cache = ToolOutputCache()
        cache.put("ReadFile", {"file_path": "a.py"}, "a")
        cache.put("GrepTool", {"pattern": "x"}, "grep result")
        cache.invalidate(tool_name="ReadFile")
        assert cache.get("ReadFile", {"file_path": "a.py"}) is None
        assert cache.get("GrepTool", {"pattern": "x"}) == "grep result"


class TestStats:
    """Cache statistics reporting."""

    def test_hit_rate(self, test_file: Path):
        cache = ToolOutputCache()
        cached = cache.wrap(lambda name, inp: "content")
        cached("ReadFile", {"file_path": str(test_file)})
        cached("ReadFile", {"file_path": str(test_file)})
        cached("ReadFile", {"file_path": str(test_file)})
        stats = cache.stats
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert abs(stats["hit_rate"] - 2 / 3) < 0.01
