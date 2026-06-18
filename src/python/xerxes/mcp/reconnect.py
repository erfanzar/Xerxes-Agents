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
"""Exponential-backoff reconnect for flaky MCP connections.

MCP servers are subprocesses or remote endpoints that occasionally drop.
:func:`reconnect_with_backoff` wraps an arbitrary connect callable in a
retry loop (5 attempts at 1s, 2s, 4s, 8s, 16s by default).
Exception messages are passed through :func:`scrub_credentials` so leaked
API keys / bearer tokens don't end up in logs or the TUI.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


class MCPReconnectError(RuntimeError):
    """Raised when :func:`reconnect_with_backoff` exhausts all attempts.

    Wraps the final underlying failure with a single, credential-scrubbed
    message. The original exception is preserved via ``__cause__`` (``raise
    ... from``). Using a dedicated single-argument exception avoids trying to
    reconstruct the original type, which fails for exceptions whose
    constructor is not ``(str) -> exc`` (e.g. ``ExceptionGroup`` /
    ``BaseExceptionGroup`` raised by anyio task groups in the MCP SDK
    transports).
    """


@dataclass
class ReconnectPolicy:
    """Exponential-backoff parameters for :func:`reconnect_with_backoff`.

    Attributes:
        max_attempts: Total attempts before giving up.
        base_seconds: Delay before the second attempt.
        factor: Multiplier applied after every failed attempt.
        max_seconds: Upper bound on any single delay.
    """

    max_attempts: int = 5
    base_seconds: float = 1.0
    factor: float = 2.0
    max_seconds: float = 60.0


_CRED_PATTERNS = [
    re.compile(r"(api[_-]?key)[\s:=\"']+([A-Za-z0-9._\-]{8,})", re.I),
    re.compile(r"(token)[\s:=\"']+([A-Za-z0-9._\-]{16,})", re.I),
    re.compile(r"(authorization:\s*bearer)\s+([A-Za-z0-9._\-]+)", re.I),
    re.compile(r"(password)[\s:=\"']+(\S+)", re.I),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
]


def scrub_credentials(text: str) -> str:
    """Replace api-key / bearer-token / password fragments in ``text`` with ``[redacted]``."""
    out = text
    for pattern in _CRED_PATTERNS:
        # Simple ``re.sub`` with a function that preserves the label half.
        out = pattern.sub(lambda m: f"{m.group(1)}=[redacted]" if m.groups() else "[redacted]", out)
    return out


def reconnect_with_backoff(
    connect: Callable[[], T],
    *,
    policy: ReconnectPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_error: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Retry ``connect`` with exponential backoff until success or ``max_attempts``.

    Each failure invokes ``on_error(attempt, exc)`` if supplied; after the
    last attempt fails the most recent exception is re-raised with its
    message scrubbed of credentials. ``sleep`` is injectable so tests can
    skip real time.
    """

    p = policy or ReconnectPolicy()
    last_exc: BaseException | None = None
    for attempt in range(1, p.max_attempts + 1):
        try:
            return connect()
        except BaseException as exc:
            last_exc = exc
            if on_error is not None:
                on_error(attempt, exc)
            if attempt >= p.max_attempts:
                break
            delay = min(p.max_seconds, p.base_seconds * (p.factor ** (attempt - 1)))
            sleep(delay)
    # Re-raise with credentials scrubbed. Do not reconstruct the original
    # exception type -- many exceptions (e.g. ExceptionGroup) cannot be
    # rebuilt from a single string. Wrap in a dedicated single-message error
    # and chain the original via ``from`` so it is never lost.
    assert last_exc is not None
    scrubbed = scrub_credentials(str(last_exc))
    raise MCPReconnectError(scrubbed) from last_exc


__all__ = ["MCPReconnectError", "ReconnectPolicy", "reconnect_with_backoff", "scrub_credentials"]
