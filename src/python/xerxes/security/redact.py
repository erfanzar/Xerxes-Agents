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
"""PII / credential redaction for logs + audit events.

A small set of regex rules catches the most common leak shapes (phone
numbers, emails, API keys, JWTs, ``Authorization`` headers). The
functions are pure and deterministic so they're safe to call from
logging filter chains."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from re import Pattern
from typing import Any


@dataclass(frozen=True)
class RedactionRule:
    """A single named regex -> replacement rule.

    Attributes:
        name: rule identifier (used for diagnostics, never user-facing).
        pattern: compiled regex applied to candidate strings.
        replacement: substitution string; may reference capture groups.
    """

    name: str
    pattern: Pattern[str]
    replacement: str = "[redacted]"


def _rule(name: str, pat: str, replacement: str = "[redacted]") -> RedactionRule:
    """Helper: compile ``pat`` case-insensitively and wrap into a rule."""
    return RedactionRule(name=name, pattern=re.compile(pat, re.I), replacement=replacement)


DEFAULT_PATTERNS: tuple[RedactionRule, ...] = (
    _rule("api_key_field", r"(api[_-]?key)[\s:=\"']+([A-Za-z0-9._\-]{8,})", r"\1=[redacted]"),
    _rule("openai_token", r"sk-[A-Za-z0-9]{16,}"),
    _rule("anthropic_token", r"sk-ant-[A-Za-z0-9_\-]{16,}"),
    _rule("bearer_header", r"(authorization:\s*bearer)\s+([A-Za-z0-9._\-]+)", r"\1 [redacted]"),
    _rule("password_field", r"(password)[\s:=\"']+(\S+)", r"\1=[redacted]"),
    _rule("jwt_token", r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
    _rule("phone_us", r"\b(?:\+1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
    _rule("phone_international", r"\+\d{1,3}[\s.-]?\d{4,14}"),
    _rule("email", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    _rule("aws_access_key", r"AKIA[0-9A-Z]{16}"),
    _rule("github_pat", r"ghp_[A-Za-z0-9]{20,}"),
)


SENSITIVE_FIELD_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "token",
        "access_token",
        "refresh_token",
        "password",
        "secret",
        "client_secret",
        "authorization",
    }
)


def redact_string(text: str, *, rules: Iterable[RedactionRule] = DEFAULT_PATTERNS) -> str:
    """Apply every rule to ``text`` and return the redacted result."""
    out = text
    for rule in rules:
        if rule.pattern.groups:
            out = rule.pattern.sub(rule.replacement, out)
        else:
            out = rule.pattern.sub(rule.replacement, out)
    return out


def redact_payload(value: Any, *, rules: Iterable[RedactionRule] = DEFAULT_PATTERNS) -> Any:
    """Recursively redact strings + sensitive-named fields in a payload."""
    if isinstance(value, dict):
        return {
            k: "[redacted]" if k.lower() in SENSITIVE_FIELD_NAMES else redact_payload(v, rules=rules)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact_payload(v, rules=rules) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_payload(v, rules=rules) for v in value)
    if isinstance(value, str):
        return redact_string(value, rules=rules)
    return value


class LoggingFilter(logging.Filter):
    """Apply ``redact_string`` to every formatted log record's message."""

    def __init__(self, rules: Iterable[RedactionRule] = DEFAULT_PATTERNS) -> None:
        """Cache the rules tuple so each filter call avoids list growth."""
        super().__init__()
        self._rules = tuple(rules)

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact ``record.msg`` and string ``record.args`` in place.

        Always returns True so the record is still emitted; redaction
        failures degrade silently rather than dropping the log entry."""
        try:
            record.msg = redact_string(str(record.msg), rules=self._rules)
            if record.args:
                record.args = tuple(
                    redact_string(str(a), rules=self._rules) if isinstance(a, str) else a for a in record.args
                )
        except Exception:
            return True
        return True


__all__ = [
    "DEFAULT_PATTERNS",
    "SENSITIVE_FIELD_NAMES",
    "LoggingFilter",
    "RedactionRule",
    "redact_payload",
    "redact_string",
]
