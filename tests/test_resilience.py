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
"""Tests for BackoffPolicy and ToolResultCache."""

from __future__ import annotations

import random

from xerxes.runtime.resilience import BackoffPolicy, JitterMode, ToolResultCache, hash_args


class TestBackoff:
    def test_no_jitter_is_exponential(self):
        p = BackoffPolicy(base_delay=1.0, mode=JitterMode.NONE, multiplier=2.0)
        assert p.delay(0) == 1.0
        assert p.delay(1) == 2.0
        assert p.delay(2) == 4.0

    def test_capped_by_max_delay(self):
        p = BackoffPolicy(base_delay=1.0, max_delay=5.0, mode=JitterMode.NONE)
        assert p.delay(10) == 5.0

    def test_full_jitter_in_range(self):
        rng = random.Random(42)
        p = BackoffPolicy(base_delay=1.0, mode=JitterMode.FULL, rng=rng)
        for i in range(5):
            d = p.delay(i)
            cap = min(p.max_delay, p.base_delay * (p.multiplier**i))
            assert 0.0 <= d <= cap

    def test_equal_jitter_stays_in_band(self):
        rng = random.Random(7)
        p = BackoffPolicy(base_delay=1.0, mode=JitterMode.EQUAL, rng=rng)
        for i in range(5):
            d = p.delay(i)
            cap = min(p.max_delay, p.base_delay * (p.multiplier**i))
            assert cap / 2.0 <= d <= cap + 1e-9

    def test_decorrelated_jitter_uses_last(self):
        rng = random.Random(99)
        p = BackoffPolicy(base_delay=1.0, mode=JitterMode.DECORRELATED, rng=rng)
        delays = list(p.sleep_iter(10))
        assert all(d <= p.max_delay for d in delays)

    def test_sleep_iter_yields_n_items(self):
        p = BackoffPolicy(base_delay=0.01, mode=JitterMode.NONE)
        assert len(list(p.sleep_iter(7))) == 7


class TestToolResultCache:
    def test_miss_then_hit(self):
        c = ToolResultCache(ttl_seconds=10)
        hit, val = c.get("Read", {"x": 1})
        assert hit is False
        c.set("Read", {"x": 1}, "abc")
        hit, val = c.get("Read", {"x": 1})
        assert hit and val == "abc"

    def test_get_or_set_computes_once(self):
        c = ToolResultCache(ttl_seconds=10)
        calls = {"n": 0}

        def producer():
            calls["n"] += 1
            return "v"

        hit, v = c.get_or_set("X", {"a": 1}, producer)
        assert hit is False and v == "v"
        hit, v = c.get_or_set("X", {"a": 1}, producer)
        assert hit is True and v == "v"
        assert calls["n"] == 1

    def test_ttl_expiry(self):
        c = ToolResultCache(ttl_seconds=5)
        c.set("X", {"a": 1}, "v", now=0)
        hit, _ = c.get("X", {"a": 1}, now=2)
        assert hit is True
        hit, _ = c.get("X", {"a": 1}, now=10)
        assert hit is False

    def test_lru_eviction(self):
        c = ToolResultCache(ttl_seconds=100, max_entries=3)
        c.set("A", {}, 1)
        c.set("B", {}, 2)
        c.set("C", {}, 3)
        c.get("A", {})  # touch A
        c.set("D", {}, 4)  # evicts B (oldest)
        assert c.get("A", {})[0] is True
        assert c.get("B", {})[0] is False
        assert c.get("C", {})[0] is True
        assert c.get("D", {})[0] is True

    def test_invalidate_specific_tool(self):
        c = ToolResultCache()
        c.set("A", {"x": 1}, 1)
        c.set("A", {"x": 2}, 2)
        c.set("B", {"x": 1}, 3)
        n = c.invalidate("A")
        assert n == 2
        assert c.get("B", {"x": 1})[0] is True

    def test_invalidate_all(self):
        c = ToolResultCache()
        c.set("A", {}, 1)
        c.set("B", {}, 2)
        n = c.invalidate()
        assert n == 2
        assert len(c) == 0

    def test_hits_misses_counters(self):
        c = ToolResultCache()
        c.get("X", {"a": 1})
        c.set("X", {"a": 1}, 1)
        c.get("X", {"a": 1})
        assert c.hits == 1
        assert c.misses == 1

    def test_arg_hash_stable_across_key_order(self):
        a = hash_args({"a": 1, "b": 2})
        b = hash_args({"b": 2, "a": 1})
        assert a == b

    def test_threadsafe(self):
        import threading

        c = ToolResultCache(ttl_seconds=10, max_entries=10000)

        def worker(i):
            for j in range(100):
                c.set(f"T{i}", {"j": j}, j)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(c) <= 10000
