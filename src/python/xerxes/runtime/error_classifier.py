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
"""Classify LLM-API errors into recovery-oriented categories.

Replaces ad-hoc ``isinstance`` chains in :mod:`xerxes.runtime.fallback` with
a small typed pipeline. Each :class:`ErrorKind` maps to a recovery strategy
(retry, switch provider, wait, give up). Classification uses built-in
exception types first, then regex patterns against ``str(exc)`` so the
classifier still works when provider SDKs aren't importable.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass


class ErrorKind(enum.Enum):
    """Recovery-shaped categorisation of an LLM-API error.

    Attributes:
        RATE_LIMIT: HTTP 429 or provider rate-limit signal.
        CONTEXT_OVERFLOW: Prompt exceeded the model's context window.
        PROVIDER_DOWN: 5xx, "overloaded", or other transient provider failure.
        AUTH: 401 / invalid API key.
        QUOTA_EXCEEDED: Billing or credit exhaustion.
        TIMEOUT: Network or request timeout (408 / ``TimeoutError``).
        TRANSIENT: Catch-all for transient retryable issues.
        BAD_REQUEST: 400 / malformed payload — won't succeed on retry.
        FATAL: Non-recoverable (e.g. user interrupt).
        UNKNOWN: Classifier had no opinion.
    """

    RATE_LIMIT = "rate_limit"
    CONTEXT_OVERFLOW = "context_overflow"
    PROVIDER_DOWN = "provider_down"
    AUTH = "auth"
    QUOTA_EXCEEDED = "quota_exceeded"
    TIMEOUT = "timeout"
    TRANSIENT = "transient"
    BAD_REQUEST = "bad_request"
    FATAL = "fatal"
    UNKNOWN = "unknown"


@dataclass
class ClassifiedError:
    """Result of classifying an exception via :class:`ErrorClassifier`.

    Attributes:
        kind: Recovery-shaped category for ``original``.
        retryable: Whether a retry could plausibly succeed.
        original: The underlying exception.
        message: Short human description, typically ``str(original)``.
        suggested_backoff_seconds: Hint extracted from ``Retry-After``-style
            text, or ``None`` when no hint was found.
    """

    kind: ErrorKind
    retryable: bool
    original: BaseException
    message: str = ""
    suggested_backoff_seconds: float | None = None


# String patterns that identify each kind. Used as a fallback when
# ``isinstance`` checks against the provider SDKs fail (e.g. when the
# SDK isn't importable in the current environment).
_PATTERNS: dict[ErrorKind, list[re.Pattern[str]]] = {
    ErrorKind.RATE_LIMIT: [
        re.compile(r"rate.?limit", re.I),
        re.compile(r"too many requests", re.I),
        re.compile(r"\b429\b"),
    ],
    ErrorKind.CONTEXT_OVERFLOW: [
        re.compile(r"context.{0,8}length", re.I),
        re.compile(r"maximum.{0,8}context", re.I),
        re.compile(r"context window", re.I),
        re.compile(r"too many tokens", re.I),
        re.compile(r"reduce.{0,8}messages", re.I),
    ],
    ErrorKind.AUTH: [
        re.compile(r"unauthorized", re.I),
        re.compile(r"invalid.{0,4}api.{0,4}key", re.I),
        re.compile(r"\b401\b"),
    ],
    ErrorKind.QUOTA_EXCEEDED: [
        re.compile(r"quota", re.I),
        re.compile(r"insufficient.{0,8}credit", re.I),
        re.compile(r"billing", re.I),
    ],
    ErrorKind.PROVIDER_DOWN: [
        re.compile(r"\b50[023]\b"),  # 500, 502, 503
        re.compile(r"service unavailable", re.I),
        re.compile(r"overloaded", re.I),
        re.compile(r"bad gateway", re.I),
    ],
    ErrorKind.TIMEOUT: [
        re.compile(r"timeout", re.I),
        re.compile(r"timed out", re.I),
        re.compile(r"\b408\b"),
    ],
    ErrorKind.BAD_REQUEST: [
        re.compile(r"\b400\b"),
        re.compile(r"invalid request", re.I),
        re.compile(r"malformed", re.I),
    ],
}

_RETRYABLE: frozenset[ErrorKind] = frozenset(
    {
        ErrorKind.RATE_LIMIT,
        ErrorKind.PROVIDER_DOWN,
        ErrorKind.TIMEOUT,
        ErrorKind.TRANSIENT,
    }
)


def _parse_retry_after(msg: str) -> float | None:
    """Look for ``Retry-After: N`` or ``retry after N seconds`` hints."""
    m = re.search(r"retry[- ]after[:\s]+(\d+(?:\.\d+)?)", msg, re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


class ErrorClassifier:
    """Stateless classifier. Instances exist mainly for future per-provider rules."""

    def classify(self, exc: BaseException) -> ClassifiedError:
        """Categorise ``exc`` and return a :class:`ClassifiedError`."""
        msg = str(exc) or exc.__class__.__name__

        # Built-in exception checks (no SDK imports required).
        if isinstance(exc, TimeoutError):
            return ClassifiedError(
                kind=ErrorKind.TIMEOUT,
                retryable=True,
                original=exc,
                message=msg,
                suggested_backoff_seconds=_parse_retry_after(msg),
            )
        if isinstance(exc, ConnectionError):
            return ClassifiedError(
                kind=ErrorKind.PROVIDER_DOWN,
                retryable=True,
                original=exc,
                message=msg,
            )
        if isinstance(exc, KeyboardInterrupt):
            return ClassifiedError(
                kind=ErrorKind.FATAL,
                retryable=False,
                original=exc,
                message="user interrupt",
            )

        # Pattern match against the message text.
        backoff = _parse_retry_after(msg)
        for kind, patterns in _PATTERNS.items():
            for pat in patterns:
                if pat.search(msg):
                    return ClassifiedError(
                        kind=kind,
                        retryable=kind in _RETRYABLE,
                        original=exc,
                        message=msg,
                        suggested_backoff_seconds=backoff,
                    )

        return ClassifiedError(
            kind=ErrorKind.UNKNOWN,
            retryable=False,
            original=exc,
            message=msg,
            suggested_backoff_seconds=backoff,
        )

    def is_retryable(self, exc: BaseException) -> bool:
        """Return ``True`` when :meth:`classify` deems ``exc`` retryable."""
        return self.classify(exc).retryable


_default_classifier = ErrorClassifier()


def classify(exc: BaseException) -> ClassifiedError:
    """Classify ``exc`` using the module-level default :class:`ErrorClassifier`."""
    return _default_classifier.classify(exc)


__all__ = ["ClassifiedError", "ErrorClassifier", "ErrorKind", "classify"]
