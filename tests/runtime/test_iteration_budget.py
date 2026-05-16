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
"""Tests for xerxes.runtime.iteration_budget."""

from __future__ import annotations

import threading

import pytest
from xerxes.runtime.iteration_budget import BudgetExhausted, IterationBudget


class TestIterationBudget:
    def test_default_max(self) -> None:
        b = IterationBudget()
        assert b.max_iterations == 50
        assert b.remaining == 50
        assert b.used == 0
        assert not b.exhausted

    def test_consume_decrements(self) -> None:
        b = IterationBudget(max_iterations=5)
        b.consume()
        assert b.used == 1
        assert b.remaining == 4

    def test_consume_multi(self) -> None:
        b = IterationBudget(max_iterations=10)
        b.consume(3)
        assert b.used == 3

    def test_consume_zero_raises(self) -> None:
        b = IterationBudget()
        with pytest.raises(ValueError):
            b.consume(0)

    def test_consume_overflow_raises(self) -> None:
        b = IterationBudget(max_iterations=2)
        b.consume(2)
        with pytest.raises(BudgetExhausted):
            b.consume()
        assert b.exhausted

    def test_try_consume_returns_bool(self) -> None:
        b = IterationBudget(max_iterations=1)
        assert b.try_consume() is True
        assert b.try_consume() is False

    def test_refund(self) -> None:
        b = IterationBudget(max_iterations=10)
        b.consume(5)
        b.refund(3)
        assert b.used == 2

    def test_refund_clamps_at_zero(self) -> None:
        b = IterationBudget()
        b.consume()
        b.refund(10)
        assert b.used == 0

    def test_refund_zero_raises(self) -> None:
        b = IterationBudget()
        with pytest.raises(ValueError):
            b.refund(0)

    def test_reset(self) -> None:
        b = IterationBudget(max_iterations=10)
        b.consume(5)
        b.reset()
        assert b.used == 0

    def test_thread_safety(self) -> None:
        b = IterationBudget(max_iterations=1000)
        threads = []

        def hit() -> None:
            for _ in range(100):
                b.try_consume()

        for _ in range(10):
            t = threading.Thread(target=hit)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert b.used == 1000
