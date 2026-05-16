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
"""Tests for xerxes.runtime.interrupt."""

from __future__ import annotations

import threading

import pytest
from xerxes.runtime.interrupt import (
    InterruptToken,
    clear_current_token,
    current_token,
    interrupt_scope,
    set_current_token,
)


class TestInterruptToken:
    def test_initial_state(self) -> None:
        t = InterruptToken()
        assert t.is_set() is False

    def test_set_and_is_set(self) -> None:
        t = InterruptToken()
        t.set()
        assert t.is_set() is True

    def test_clear(self) -> None:
        t = InterruptToken()
        t.set()
        t.clear()
        assert t.is_set() is False

    def test_raise_if_set(self) -> None:
        t = InterruptToken()
        t.raise_if_set()  # no-op when clear
        t.set()
        with pytest.raises(KeyboardInterrupt):
            t.raise_if_set()

    def test_wait_returns_true_on_set(self) -> None:
        t = InterruptToken()
        threading.Timer(0.05, t.set).start()
        assert t.wait(timeout=1.0) is True

    def test_wait_timeout_returns_false(self) -> None:
        t = InterruptToken()
        assert t.wait(timeout=0.01) is False


class TestThreadLocalToken:
    def teardown_method(self) -> None:
        clear_current_token()

    def test_current_token_default_none(self) -> None:
        assert current_token() is None

    def test_set_and_get(self) -> None:
        t = InterruptToken()
        set_current_token(t)
        assert current_token() is t

    def test_clear(self) -> None:
        set_current_token(InterruptToken())
        clear_current_token()
        assert current_token() is None

    def test_isolation_across_threads(self) -> None:
        main_token = InterruptToken()
        set_current_token(main_token)

        seen: dict[str, InterruptToken | None] = {}

        def worker() -> None:
            seen["worker"] = current_token()

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        # Worker sees a different (None) value.
        assert seen["worker"] is None
        # Main thread still sees its token.
        assert current_token() is main_token


class TestInterruptScope:
    def teardown_method(self) -> None:
        clear_current_token()

    def test_installs_token_inside_block(self) -> None:
        with interrupt_scope() as tok:
            assert current_token() is tok

    def test_restores_previous_token(self) -> None:
        outer = InterruptToken()
        set_current_token(outer)
        with interrupt_scope():
            pass
        assert current_token() is outer

    def test_accepts_explicit_token(self) -> None:
        passed = InterruptToken()
        with interrupt_scope(passed) as tok:
            assert tok is passed
            assert current_token() is passed

    def test_nested_scopes(self) -> None:
        with interrupt_scope() as outer_tok:
            with interrupt_scope() as inner_tok:
                assert current_token() is inner_tok
                assert inner_tok is not outer_tok
            assert current_token() is outer_tok
