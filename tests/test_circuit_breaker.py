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
"""Tests for CircuitBreakerRegistry."""

from __future__ import annotations

import pytest
from xerxes.runtime.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
)


@pytest.fixture
def br():
    return CircuitBreakerRegistry(CircuitBreakerConfig(failure_threshold=3, cooldown_seconds=10))


class TestBreaker:
    def test_starts_closed(self, br):
        assert br.state_of("x") == CircuitState.CLOSED
        assert br.should_allow("x") is True

    def test_trips_after_threshold_failures(self, br):
        for _ in range(2):
            br.record_failure("x")
        assert br.state_of("x") == CircuitState.CLOSED
        br.record_failure("x")
        assert br.state_of("x") == CircuitState.OPEN
        assert br.should_allow("x") is False

    def test_half_open_after_cooldown(self):
        br = CircuitBreakerRegistry(CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=10))
        br.record_failure("x", now=0)
        assert br.state_of("x") == CircuitState.OPEN
        assert br.should_allow("x", now=15) is True
        assert br.state_of("x") == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self):
        br = CircuitBreakerRegistry(CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=10, success_threshold=1))
        br.record_failure("x", now=0)
        br.should_allow("x", now=15)
        br.record_success("x")
        assert br.state_of("x") == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        br = CircuitBreakerRegistry(CircuitBreakerConfig(failure_threshold=1, cooldown_seconds=10))
        br.record_failure("x", now=0)
        br.should_allow("x", now=15)
        tripped = br.record_failure("x", now=15)
        assert tripped is True
        assert br.state_of("x") == CircuitState.OPEN

    def test_rolling_window_forgets_old_failures(self):
        br = CircuitBreakerRegistry(
            CircuitBreakerConfig(failure_threshold=3, cooldown_seconds=10, rolling_window_seconds=5)
        )
        br.record_failure("x", now=0)
        br.record_failure("x", now=1)
        br.record_failure("x", now=10)
        assert br.state_of("x") == CircuitState.CLOSED

    def test_call_wraps_callable(self, br):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            return "ok"

        assert br.call("x", fn) == "ok"
        assert calls["n"] == 1

    def test_call_records_failure_on_exception(self, br):
        def fn():
            raise ValueError("boom")

        for _ in range(3):
            with pytest.raises(ValueError):
                br.call("x", fn)
        assert br.state_of("x") == CircuitState.OPEN
        with pytest.raises(CircuitOpenError):
            br.call("x", fn)

    def test_reset_specific(self, br):
        br.record_failure("x")
        br.record_failure("y")
        br.reset("x")
        assert br.state_of("x") == CircuitState.CLOSED

    def test_reset_all(self, br):
        for _ in range(3):
            br.record_failure("x")
        br.reset()
        assert br.state_of("x") == CircuitState.CLOSED

    def test_threadsafe(self, br):
        import threading

        def hammer():
            for _ in range(100):
                br.record_failure("x")

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert br.state_of("x") == CircuitState.OPEN
