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
"""Tests for FallbackRegistry and ToolHealthProber."""

from __future__ import annotations

from xerxes.runtime.fallback import (
    FallbackRegistry,
    HealthSnapshot,
    ToolHealthProber,
)


class TestFallback:
    def test_set_and_order(self):
        r = FallbackRegistry()
        r.set("sum", "opus", alternatives=["sonnet", "haiku"])
        assert r.order_for("sum") == ["opus", "sonnet", "haiku"]

    def test_dedup_in_order(self):
        r = FallbackRegistry()
        r.set("sum", "opus", alternatives=["opus", "haiku", "haiku"])
        assert r.order_for("sum") == ["opus", "haiku"]

    def test_next_after(self):
        r = FallbackRegistry()
        r.set("sum", "opus", alternatives=["sonnet", "haiku"])
        assert r.next_after("sum", "opus") == "sonnet"
        assert r.next_after("sum", "sonnet") == "haiku"
        assert r.next_after("sum", "haiku") is None

    def test_unknown_capability_returns_empty(self):
        r = FallbackRegistry()
        assert r.order_for("nope") == []
        assert r.next_after("nope", "x") is None

    def test_remove(self):
        r = FallbackRegistry()
        r.set("c", "a")
        assert r.remove("c") is True
        assert r.remove("c") is False


class TestProber:
    def test_register_and_run_one(self):
        p = ToolHealthProber()
        p.register("ping", lambda: True)
        snap = p.run_one("ping")
        assert snap.status == "ok"
        assert snap.latency_ms >= 0

    def test_returning_false_means_down(self):
        p = ToolHealthProber()
        p.register("svc", lambda: False)
        assert p.run_one("svc").status == "down"

    def test_exception_is_down_with_message(self):
        p = ToolHealthProber()

        def boom():
            raise RuntimeError("api 503")

        p.register("svc", boom)
        snap = p.run_one("svc")
        assert snap.status == "down"
        assert "api 503" in snap.message

    def test_explicit_snapshot_returned_directly(self):
        p = ToolHealthProber()

        def detail():
            return HealthSnapshot(name="svc", status="degraded", message="latency rising")

        p.register("svc", detail)
        assert p.run_one("svc").status == "degraded"

    def test_run_due_only_runs_overdue(self):
        p = ToolHealthProber()
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            return True

        p.register("svc", fn, interval_seconds=10)
        p.run_due(now=0)
        p.run_due(now=1)
        assert calls["n"] == 1
        p.run_due(now=11)
        assert calls["n"] == 2

    def test_unknown_probe(self):
        p = ToolHealthProber()
        snap = p.run_one("nope")
        assert snap.status == "unknown"

    def test_unregister(self):
        p = ToolHealthProber()
        p.register("x", lambda: True, interval_seconds=1)
        p.run_one("x")
        p.unregister("x")
        assert p.snapshot("x") is None

    def test_healthy_helper(self):
        p = ToolHealthProber()
        p.register("a", lambda: True)
        p.register("b", lambda: False)
        p.run_due(now=0)
        assert p.healthy("a") is True
        assert p.healthy("b") is False
