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
"""Tests for redact / url_safety / path_security / approvals."""

from __future__ import annotations

import logging

import pytest
from xerxes.security.approvals import ApprovalRecord, ApprovalScope, ApprovalStore
from xerxes.security.path_security import PathEscape, resolve_within, safe_path
from xerxes.security.redact import LoggingFilter, redact_payload, redact_string
from xerxes.security.url_safety import check_url, is_url_safe

# ---------------------------- redact ---------------------------------------


class TestRedactString:
    def test_openai_token_redacted(self):
        out = redact_string("oops sk-abcdefghij1234567890")
        assert "sk-abcdefghij" not in out

    def test_anthropic_token_redacted(self):
        out = redact_string("sk-ant-abc123def456ghi789")
        assert "ant-" not in out

    def test_email_redacted(self):
        out = redact_string("contact us at foo@example.com")
        assert "foo@example.com" not in out

    def test_phone_us_redacted(self):
        out = redact_string("call me at 555-123-4567")
        assert "555-123-4567" not in out

    def test_aws_key_redacted(self):
        out = redact_string("AKIAIOSFODNN7EXAMPLE in logs")
        assert "AKIAIOSFODNN7EXAMPLE" not in out

    def test_bearer_redacted(self):
        out = redact_string("Authorization: Bearer abc123XYZ789LONGTOKEN")
        assert "abc123XYZ789LONGTOKEN" not in out


class TestRedactPayload:
    def test_sensitive_field_blanked(self):
        out = redact_payload({"username": "alice", "password": "hunter2"})
        assert out["password"] == "[redacted]"
        assert out["username"] == "alice"

    def test_nested(self):
        out = redact_payload({"auth": {"token": "abc", "fresh": "ok"}})
        assert out["auth"]["token"] == "[redacted]"
        assert out["auth"]["fresh"] == "ok"

    def test_string_in_list(self):
        out = redact_payload(["contact me at a@b.com"])
        assert "@b.com" not in out[0]

    def test_non_string_untouched(self):
        out = redact_payload({"count": 42, "ratio": 0.5})
        assert out == {"count": 42, "ratio": 0.5}


class TestLoggingFilter:
    def test_filter_applies_to_message(self):
        f = LoggingFilter()
        record = logging.LogRecord("x", logging.INFO, "p", 1, "key=sk-abcdefghij0123456789", None, None)
        f.filter(record)
        assert "sk-abcdefghij" not in record.msg


# ---------------------------- url_safety -----------------------------------


class TestUrlSafety:
    def test_public_url_allowed(self):
        assert is_url_safe("https://example.com/api") is True

    def test_localhost_blocked(self):
        assert is_url_safe("http://localhost:8000/internal") is False

    def test_rfc1918_blocked(self):
        assert is_url_safe("http://192.168.1.1/") is False

    def test_loopback_blocked(self):
        assert is_url_safe("http://127.0.0.1/") is False

    def test_link_local_blocked(self):
        assert is_url_safe("http://169.254.169.254/latest/meta-data/") is False

    def test_file_scheme_blocked(self):
        assert is_url_safe("file:///etc/passwd") is False

    def test_allowlist_overrides(self):
        ok = is_url_safe("http://localhost:8000/dev", allowlist=frozenset({"localhost"}))
        assert ok is True

    def test_check_url_reason(self):
        decision = check_url("http://192.168.1.1/x")
        assert decision.allowed is False
        assert "private" in decision.reason


# ---------------------------- path_security --------------------------------


class TestPathSecurity:
    def test_resolve_within_ok(self, tmp_path):
        out = resolve_within(tmp_path, "subdir/file.txt")
        assert str(out).startswith(str(tmp_path.resolve()))

    def test_escape_raises(self, tmp_path):
        with pytest.raises(PathEscape):
            resolve_within(tmp_path, "../../etc/passwd")

    def test_absolute_paths_rerooted(self, tmp_path):
        # An "absolute" path is interpreted relative to workspace.
        out = resolve_within(tmp_path, "/etc/passwd")
        assert str(out).startswith(str(tmp_path.resolve()))

    def test_safe_path_returns_none_on_escape(self, tmp_path):
        assert safe_path(tmp_path, "../escape") is None


# ---------------------------- approvals ------------------------------------


class TestApprovals:
    def test_no_match_returns_none(self):
        store = ApprovalStore()
        assert store.check(tool_name="run", session_id="s1") is None

    def test_once_scoped_match(self):
        store = ApprovalStore()
        store.add(
            ApprovalRecord(tool_name="run", scope=ApprovalScope.ONCE, granted=True, session_id="s1", args_hash="h1")
        )
        assert store.check(tool_name="run", session_id="s1", args_hash="h1") is True
        assert store.check(tool_name="run", session_id="s1", args_hash="h2") is None

    def test_session_scope(self):
        store = ApprovalStore()
        store.add(ApprovalRecord(tool_name="run", scope=ApprovalScope.SESSION, granted=True, session_id="s1"))
        assert store.check(tool_name="run", session_id="s1") is True
        assert store.check(tool_name="run", session_id="s2") is None

    def test_always_scope_persists_to_disk(self, tmp_path):
        path = tmp_path / "approvals.json"
        store = ApprovalStore(persist_path=path)
        store.add(ApprovalRecord(tool_name="run", scope=ApprovalScope.ALWAYS, granted=True))
        assert path.exists()
        # Load fresh store.
        store2 = ApprovalStore(persist_path=path)
        assert store2.check(tool_name="run", session_id="any") is True

    def test_deny_returns_false(self):
        store = ApprovalStore()
        store.add(ApprovalRecord(tool_name="rm", scope=ApprovalScope.SESSION, granted=False, session_id="s1"))
        assert store.check(tool_name="rm", session_id="s1") is False

    def test_clear_session(self):
        store = ApprovalStore()
        store.add(ApprovalRecord(tool_name="a", scope=ApprovalScope.SESSION, granted=True, session_id="s1"))
        store.add(ApprovalRecord(tool_name="b", scope=ApprovalScope.SESSION, granted=True, session_id="s2"))
        removed = store.clear_session("s1")
        assert removed == 1
        assert store.check(tool_name="a", session_id="s1") is None
        assert store.check(tool_name="b", session_id="s2") is True
