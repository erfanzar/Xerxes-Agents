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
"""Daemon session persistence — `xerxes -r <id>` rehydration regression.

The daemon used to create an empty in-memory session every time the
user passed ``-r <id>``, so resumed sessions came up with zero
history. These tests pin the new behavior: SessionManager.save writes
a session to disk; SessionManager.open rehydrates an existing record
when its id matches the requested key."""

from __future__ import annotations

import json

import pytest
from xerxes.daemon.config import DaemonConfig
from xerxes.daemon.runtime import SessionManager, WorkspaceManager, _looks_like_id


@pytest.fixture
def store(tmp_path):
    config = DaemonConfig()
    config.workspace = {"root": str(tmp_path / "agents"), "default_agent_id": "default"}
    wm = WorkspaceManager(config)
    return SessionManager(wm, store_dir=tmp_path / "sessions")


def _seed_session(store: SessionManager, key: str, *, messages=None, turns=0):
    sess = store.open(key)
    sess.state.messages = list(messages or [])
    sess.state.turn_count = turns
    sess.state.total_input_tokens = 123
    sess.state.total_output_tokens = 456
    store.save(sess)
    return sess


class TestPersistence:
    def test_save_creates_file(self, tmp_path, store):
        sess = _seed_session(store, "abc12345", messages=[{"role": "user", "content": "hi"}], turns=1)
        path = tmp_path / "sessions" / f"{sess.id}.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["messages"] == [{"role": "user", "content": "hi"}]
        assert data["turn_count"] == 1

    def test_rehydrate_on_fresh_manager(self, tmp_path, store):
        _seed_session(
            store,
            "abc12345",
            messages=[{"role": "user", "content": "remember this"}, {"role": "assistant", "content": "noted"}],
            turns=1,
        )
        # New manager (simulating daemon restart) reads from disk.
        config = DaemonConfig()
        config.workspace = {"root": str(tmp_path / "agents")}
        wm = WorkspaceManager(config)
        sm2 = SessionManager(wm, store_dir=tmp_path / "sessions")
        sess = sm2.open("abc12345")
        assert len(sess.state.messages) == 2
        assert sess.state.messages[0]["content"] == "remember this"
        assert sess.state.turn_count == 1
        assert sess.state.total_input_tokens == 123

    def test_unknown_key_creates_fresh(self, store):
        sess = store.open("never-saved")
        assert sess.state.messages == []
        assert sess.state.turn_count == 0

    def test_id_matching_key_used_as_session_id(self, store):
        # When the user types `-r abcd1234`, the daemon should use that as
        # the session id (so subsequent saves land in the right file).
        sess = store.open("abcd1234ef")
        assert sess.id == "abcd1234ef"

    def test_non_id_key_gets_uuid(self, store):
        sess = store.open("tui:default")
        # Slashes/colons make it a key, not an id — id is uuid-derived.
        assert sess.id != "tui:default"
        assert len(sess.id) == 12

    def test_save_atomic_no_temp_leftover(self, tmp_path, store):
        sess = _seed_session(store, "abc12345", messages=[{"role": "user", "content": "x"}])
        # No .tmp* files left behind.
        leftovers = [
            p.name
            for p in (tmp_path / "sessions").iterdir()
            if p.name.startswith(".") and not p.name == f"{sess.id}.json"
        ]
        assert leftovers == []

    def test_fresh_launch_does_not_rehydrate(self, tmp_path, store):
        """A new ``xerxes`` launch (no ``-r``) MUST get a clean session.

        Even when prior sessions exist on disk bound to the same slot
        key (``tui:default``), opening the key should produce an empty
        session — the user opts into resume via ``-r <id>``, never
        implicitly.
        """
        # Seed two prior sessions that share the slot key.
        _seed_session(store, "tui:default", messages=[{"role": "user", "content": "old"}])
        sessions_dir = tmp_path / "sessions"
        prior = {
            "session_id": "deadbeef",
            "key": "tui:default",
            "agent_id": "default",
            "cwd": "/tmp",
            "updated_at": "9999-01-01T00:00:00Z",
            "messages": [{"role": "user", "content": "should not appear"}],
            "turn_count": 7,
            "total_input_tokens": 1,
            "total_output_tokens": 2,
            "thinking_content": [],
            "tool_executions": [],
        }
        (sessions_dir / "deadbeef.json").write_text(json.dumps(prior))

        # New manager → fresh daemon. Opening the same slot key must NOT
        # silently surface the prior session's history.
        config = DaemonConfig()
        config.workspace = {"root": str(tmp_path / "agents")}
        wm = WorkspaceManager(config)
        sm2 = SessionManager(wm, store_dir=sessions_dir)
        sess = sm2.open("tui:default")
        assert sess.state.messages == []
        assert sess.state.turn_count == 0
        # The fresh session got its own uuid-derived id, distinct from
        # any prior session for this key.
        assert sess.id != "deadbeef"

    def test_evict_drops_cached_slot_session(self, store):
        """``evict`` removes the in-memory session so the next ``open`` is fresh.

        Regression cover for the long-lived-daemon bug: a TUI quit + relaunch
        without ``-r`` would otherwise reconnect to the same daemon and find
        the previous session still cached under ``tui:default``, replaying
        old history into the new chat.
        """
        sess1 = store.open("tui:default")
        sess1.state.messages = [{"role": "user", "content": "old"}]
        store.save(sess1)
        # Same key — should return the in-memory session with messages still
        # present (this is the cached-slot case the daemon used to leak).
        sess1b = store.open("tui:default")
        assert sess1b is sess1
        assert sess1b.state.messages == [{"role": "user", "content": "old"}]
        # After evict, the next open synthesises a brand-new empty session.
        store.evict("tui:default")
        sess2 = store.open("tui:default")
        assert sess2 is not sess1
        assert sess2.state.messages == []

    def test_corrupt_session_file_ignored(self, tmp_path, store):
        # A garbage file shouldn't crash the manager.
        sessions_dir = tmp_path / "sessions"
        (sessions_dir / "bad.json").write_text("{not valid json")
        config = DaemonConfig()
        config.workspace = {"root": str(tmp_path / "agents")}
        wm = WorkspaceManager(config)
        sm2 = SessionManager(wm, store_dir=sessions_dir)
        sess = sm2.open("never-existed")
        assert sess.state.messages == []


class TestLooksLikeId:
    def test_short_hex(self):
        assert _looks_like_id("abc12345") is True
        assert _looks_like_id("12885f1a3325") is True

    def test_too_short(self):
        assert _looks_like_id("abc") is False

    def test_has_separator(self):
        assert _looks_like_id("tui:default") is False
        assert _looks_like_id("user/agent") is False

    def test_long_uuid_form(self):
        assert _looks_like_id("abcdef0123456789abcdef0123456789") is True
