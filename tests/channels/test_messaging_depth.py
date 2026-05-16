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
"""Tests for plan 21 — sticker cache, identity hashing, session reset."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from xerxes.channels.identity_hashing import hash_chat, hash_user, matches_user
from xerxes.channels.session_reset import ResetTrigger, SessionResetPolicy, should_reset
from xerxes.channels.sticker_cache import StickerCache


class TestStickerCache:
    def test_put_and_get(self, tmp_path):
        c = StickerCache(tmp_path)
        c.put("telegram", "AgADBA", tmp_path / "a.webp")
        rec = c.get("telegram", "AgADBA")
        assert rec is not None
        assert rec.local_path.endswith("a.webp")

    def test_lru_evicts_oldest(self, tmp_path):
        c = StickerCache(tmp_path, lru_size=2)
        c.put("t", "1", tmp_path / "1.webp")
        c.put("t", "2", tmp_path / "2.webp")
        c.put("t", "3", tmp_path / "3.webp")
        assert c.get("t", "1") is None
        assert c.get("t", "2") is not None
        assert c.get("t", "3") is not None

    def test_persists_to_disk(self, tmp_path):
        c1 = StickerCache(tmp_path)
        c1.put("slack", "abc", tmp_path / "s.png")
        # New cache instance reads the index.
        c2 = StickerCache(tmp_path)
        assert c2.get("slack", "abc") is not None

    def test_clear(self, tmp_path):
        c = StickerCache(tmp_path)
        c.put("t", "1", tmp_path / "x.webp")
        c.clear()
        assert c.size() == 0


class TestIdentityHashing:
    def test_user_hash_stable(self, monkeypatch):
        monkeypatch.setenv("XERXES_IDENTITY_SALT", "test-salt")
        a = hash_user("telegram", 12345)
        b = hash_user("telegram", 12345)
        assert a == b
        assert a.startswith("user_")

    def test_user_hash_different_per_platform(self, monkeypatch):
        monkeypatch.setenv("XERXES_IDENTITY_SALT", "test-salt")
        assert hash_user("telegram", 1) != hash_user("discord", 1)

    def test_chat_hash_prefix(self, monkeypatch):
        monkeypatch.setenv("XERXES_IDENTITY_SALT", "test-salt")
        h = hash_chat("slack", "C1234")
        assert h.startswith("slack:")

    def test_matches_user(self, monkeypatch):
        monkeypatch.setenv("XERXES_IDENTITY_SALT", "test-salt")
        h = hash_user("telegram", 99)
        assert matches_user("telegram", 99, h)
        assert not matches_user("telegram", 100, h)


class TestSessionReset:
    def test_manual_only_resets_on_explicit_request(self):
        policy = SessionResetPolicy(trigger=ResetTrigger.MANUAL)
        assert should_reset(policy, last_message_at=None, message_count=1000) is False
        assert should_reset(policy, last_message_at=None, message_count=0, manual_request=True) is True

    def test_msg_count_triggers(self):
        policy = SessionResetPolicy(trigger=ResetTrigger.MSG_COUNT, msg_count=5)
        assert should_reset(policy, last_message_at=None, message_count=4) is False
        assert should_reset(policy, last_message_at=None, message_count=5) is True

    def test_timeout_triggers(self):
        policy = SessionResetPolicy(trigger=ResetTrigger.TIMEOUT, timeout_minutes=30)
        now = datetime.now(UTC)
        # Older than threshold.
        old = now - timedelta(minutes=31)
        assert should_reset(policy, last_message_at=old, message_count=0, now=now) is True
        recent = now - timedelta(minutes=10)
        assert should_reset(policy, last_message_at=recent, message_count=0, now=now) is False

    def test_timeout_with_no_last_message(self):
        policy = SessionResetPolicy(trigger=ResetTrigger.TIMEOUT, timeout_minutes=30)
        assert should_reset(policy, last_message_at=None, message_count=0) is False
