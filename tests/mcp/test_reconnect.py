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
"""Tests for xerxes.mcp.reconnect."""

from __future__ import annotations

import pytest
from xerxes.mcp.reconnect import ReconnectPolicy, reconnect_with_backoff, scrub_credentials


class TestScrubCredentials:
    def test_api_key(self):
        out = scrub_credentials('failed: api_key="sk-1234567890abcdef"')
        assert "sk-1234567890abcdef" not in out
        assert "[redacted]" in out

    def test_bearer_token(self):
        out = scrub_credentials("Authorization: Bearer abc123xyz789ZZZZZZZZ")
        assert "abc123" not in out

    def test_sk_pattern(self):
        out = scrub_credentials("oops sk-abcdefghijklmnopqrstuv leaked")
        assert "sk-abcdefghij" not in out


class TestReconnect:
    def test_succeeds_first_try(self):
        attempts = []

        def connect():
            attempts.append(1)
            return "ok"

        assert reconnect_with_backoff(connect, sleep=lambda s: None) == "ok"
        assert len(attempts) == 1

    def test_succeeds_after_retries(self):
        attempts = []

        def connect():
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("temporary")
            return "ok"

        out = reconnect_with_backoff(connect, policy=ReconnectPolicy(max_attempts=5), sleep=lambda s: None)
        assert out == "ok"
        assert len(attempts) == 3

    def test_gives_up_after_max_attempts(self):
        def connect():
            raise ConnectionError("nope")

        with pytest.raises(ConnectionError):
            reconnect_with_backoff(connect, policy=ReconnectPolicy(max_attempts=2), sleep=lambda s: None)

    def test_on_error_callback(self):
        seen = []

        def connect():
            raise ConnectionError('api_key="sk-secretkey-do-not-log"')

        def on_err(attempt, exc):
            seen.append((attempt, type(exc).__name__))

        with pytest.raises(ConnectionError) as exc_info:
            reconnect_with_backoff(
                connect, policy=ReconnectPolicy(max_attempts=2), sleep=lambda s: None, on_error=on_err
            )
        assert len(seen) == 2
        # Credentials must be scrubbed from the final raised message.
        assert "sk-secretkey" not in str(exc_info.value)

    def test_backoff_delays(self):
        delays = []

        def connect():
            raise ConnectionError("x")

        def sleep(s):
            delays.append(s)

        with pytest.raises(ConnectionError):
            reconnect_with_backoff(
                connect,
                policy=ReconnectPolicy(max_attempts=4, base_seconds=1, factor=2, max_seconds=10),
                sleep=sleep,
            )
        # 4 attempts → 3 sleeps; delays follow 1, 2, 4.
        assert delays == [1.0, 2.0, 4.0]
