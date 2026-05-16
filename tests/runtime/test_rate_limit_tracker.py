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
"""Tests for xerxes.runtime.rate_limit_tracker."""

from __future__ import annotations

import pytest
from xerxes.runtime.rate_limit_tracker import RateLimitTracker


class TestRateLimitTracker:
    def test_invalid_throttle_ratio(self) -> None:
        with pytest.raises(ValueError):
            RateLimitTracker(throttle_ratio=0)
        with pytest.raises(ValueError):
            RateLimitTracker(throttle_ratio=1)

    def test_no_state_no_throttle(self) -> None:
        t = RateLimitTracker()
        assert t.should_throttle("openai", "gpt-4o") is False
        assert t.delay_ms("openai", "gpt-4o") == 0

    def test_update_records_headers(self) -> None:
        t = RateLimitTracker()
        state = t.update(
            "openai",
            "gpt-4o",
            {
                "x-ratelimit-limit-requests": "100",
                "x-ratelimit-remaining-requests": "97",
                "x-ratelimit-limit-tokens": "100000",
                "x-ratelimit-remaining-tokens": "50000",
            },
        )
        assert state.limit_requests == 100
        assert state.remaining_requests == 97
        assert state.limit_tokens == 100000
        assert state.remaining_tokens == 50000

    def test_throttle_when_request_budget_low(self) -> None:
        t = RateLimitTracker(throttle_ratio=0.1)
        t.update(
            "openai",
            "gpt-4o",
            {"x-ratelimit-limit-requests": "100", "x-ratelimit-remaining-requests": "5"},
        )
        assert t.should_throttle("openai", "gpt-4o") is True

    def test_throttle_when_token_budget_low(self) -> None:
        t = RateLimitTracker(throttle_ratio=0.1)
        t.update(
            "anthropic",
            "claude-opus-4-7",
            {"x-ratelimit-limit-tokens": "100000", "x-ratelimit-remaining-tokens": "500"},
        )
        assert t.should_throttle("anthropic", "claude-opus-4-7") is True

    def test_retry_after_signals_throttle(self) -> None:
        t = RateLimitTracker()
        t.update("openai", "gpt-4o", {"retry-after": "30"})
        assert t.should_throttle("openai", "gpt-4o") is True
        # Suggested delay roughly equals retry-after.
        delay = t.delay_ms("openai", "gpt-4o")
        assert 28_000 <= delay <= 31_000

    def test_case_insensitive_headers(self) -> None:
        t = RateLimitTracker()
        t.update("openai", "gpt-4o", {"X-RateLimit-Limit-Requests": "100"})
        assert t.state("openai", "gpt-4o").limit_requests == 100

    def test_clear(self) -> None:
        t = RateLimitTracker()
        t.update("openai", "gpt-4o", {"x-ratelimit-limit-requests": "10"})
        t.clear()
        assert t.state("openai", "gpt-4o") is None

    def test_delay_ms_no_throttle_returns_zero(self) -> None:
        t = RateLimitTracker(throttle_ratio=0.05)
        t.update(
            "openai",
            "gpt-4o",
            {"x-ratelimit-limit-requests": "100", "x-ratelimit-remaining-requests": "80"},
        )
        assert t.delay_ms("openai", "gpt-4o") == 0
