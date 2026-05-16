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
"""Tests for xerxes.runtime.error_classifier."""

from __future__ import annotations

from xerxes.runtime.error_classifier import (
    ErrorClassifier,
    ErrorKind,
    classify,
)


class TestClassify:
    def test_timeout_builtin(self) -> None:
        out = classify(TimeoutError("read timed out"))
        assert out.kind is ErrorKind.TIMEOUT
        assert out.retryable is True

    def test_connection_error_builtin(self) -> None:
        out = classify(ConnectionError("connection refused"))
        assert out.kind is ErrorKind.PROVIDER_DOWN
        assert out.retryable is True

    def test_keyboard_interrupt_fatal(self) -> None:
        out = classify(KeyboardInterrupt())
        assert out.kind is ErrorKind.FATAL
        assert out.retryable is False

    def test_rate_limit_429(self) -> None:
        out = classify(Exception("HTTP 429 Too Many Requests"))
        assert out.kind is ErrorKind.RATE_LIMIT
        assert out.retryable is True

    def test_rate_limit_phrase(self) -> None:
        out = classify(Exception("anthropic rate-limited"))
        assert out.kind is ErrorKind.RATE_LIMIT

    def test_context_overflow(self) -> None:
        out = classify(Exception("The maximum context length is 128000 tokens"))
        assert out.kind is ErrorKind.CONTEXT_OVERFLOW
        assert out.retryable is False

    def test_unauthorized(self) -> None:
        out = classify(Exception("401 Unauthorized: invalid API key"))
        assert out.kind is ErrorKind.AUTH

    def test_quota_exceeded(self) -> None:
        out = classify(Exception("Insufficient credit on your account"))
        assert out.kind is ErrorKind.QUOTA_EXCEEDED

    def test_provider_503(self) -> None:
        out = classify(Exception("HTTP 503 Service Unavailable"))
        assert out.kind is ErrorKind.PROVIDER_DOWN
        assert out.retryable is True

    def test_bad_request_400(self) -> None:
        out = classify(Exception("HTTP 400 Invalid request"))
        assert out.kind is ErrorKind.BAD_REQUEST

    def test_unknown_default(self) -> None:
        out = classify(Exception("unrelated message"))
        assert out.kind is ErrorKind.UNKNOWN
        assert out.retryable is False

    def test_retry_after_parsed(self) -> None:
        out = classify(Exception("rate limited; retry after 12 seconds"))
        assert out.kind is ErrorKind.RATE_LIMIT
        assert out.suggested_backoff_seconds == 12.0


class TestClassifierInstance:
    def test_is_retryable_shortcut(self) -> None:
        c = ErrorClassifier()
        assert c.is_retryable(TimeoutError()) is True
        assert c.is_retryable(KeyboardInterrupt()) is False
