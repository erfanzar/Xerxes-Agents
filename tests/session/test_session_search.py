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
"""Tests for SessionIndex and SessionStore.search()."""

from __future__ import annotations

from datetime import datetime

from xerxes.memory import HashEmbedder
from xerxes.session.index import SessionIndex
from xerxes.session.models import SessionRecord, TurnRecord
from xerxes.session.store import InMemorySessionStore


def _session(sid: str, turns: list[TurnRecord]) -> SessionRecord:
    return SessionRecord(session_id=sid, created_at=datetime.now().isoformat(), turns=turns)


def _turn(turn_id: str, prompt: str, response: str = "", agent_id: str | None = None) -> TurnRecord:
    return TurnRecord(
        turn_id=turn_id,
        agent_id=agent_id,
        prompt=prompt,
        response_content=response,
        started_at=datetime.now().isoformat(),
    )


class TestSessionIndex:
    def test_index_and_search_single_turn(self):
        idx = SessionIndex(":memory:")
        sess = _session("s1", [_turn("t1", "how do I configure CI", "use github actions")])
        idx.index_session(sess)
        hits = idx.search("github actions", k=5)
        assert len(hits) == 1
        assert hits[0].turn_id == "t1"

    def test_search_filter_by_agent(self):
        idx = SessionIndex(":memory:")
        sess = _session(
            "s1",
            [
                _turn("t1", "fix login bug", agent_id="coder"),
                _turn("t2", "fix login bug", agent_id="reviewer"),
            ],
        )
        idx.index_session(sess)
        hits = idx.search("login", k=5, agent_id="coder")
        assert len(hits) == 1
        assert hits[0].agent_id == "coder"

    def test_search_filter_by_session(self):
        idx = SessionIndex(":memory:")
        idx.index_session(_session("s1", [_turn("t1", "alpha")]))
        idx.index_session(_session("s2", [_turn("t2", "alpha")]))
        hits = idx.search("alpha", k=5, session_id="s2")
        assert {h.session_id for h in hits} == {"s2"}

    def test_remove_session(self):
        idx = SessionIndex(":memory:")
        idx.index_session(_session("s1", [_turn("t1", "alpha")]))
        idx.index_session(_session("s2", [_turn("t2", "alpha")]))
        n = idx.remove_session("s1")
        assert n == 1
        hits = idx.search("alpha", k=5)
        assert {h.session_id for h in hits} == {"s2"}

    def test_index_replace_keeps_one_row(self):
        idx = SessionIndex(":memory:")
        sess = _session("s1", [_turn("t1", "first version")])
        idx.index_session(sess)
        sess.turns[0].prompt = "second version"
        idx.index_session(sess)
        hits = idx.search("second", k=5)
        assert len(hits) == 1
        hits = idx.search("first", k=5)
        assert hits == []

    def test_empty_query_returns_empty(self):
        idx = SessionIndex(":memory:")
        idx.index_session(_session("s1", [_turn("t1", "anything")]))
        assert idx.search("") == []

    def test_hybrid_search_with_embedder(self):
        idx = SessionIndex(":memory:", embedder=HashEmbedder())
        idx.index_session(
            _session(
                "s1",
                [
                    _turn("t1", "set up continuous integration with github actions"),
                    _turn("t2", "make me a sandwich"),
                ],
            )
        )
        hits = idx.search("github actions", k=2)
        assert hits[0].turn_id == "t1"
        assert hits[0].score > 0


class TestSessionStoreSearch:
    def test_in_memory_store_search(self):
        store = InMemorySessionStore()
        store.save_session(_session("s1", [_turn("t1", "hello world")]))
        store.save_session(_session("s2", [_turn("t2", "anything else")]))
        hits = store.search("hello", k=5)
        assert len(hits) == 1
        assert hits[0].session_id == "s1"

    def test_store_search_empty_query(self):
        store = InMemorySessionStore()
        store.save_session(_session("s1", [_turn("t1", "alpha")]))
        assert store.search("") == []

    def test_store_search_filter_agent(self):
        store = InMemorySessionStore()
        store.save_session(
            _session(
                "s1",
                [
                    _turn("t1", "X", agent_id="a"),
                    _turn("t2", "X", agent_id="b"),
                ],
            )
        )
        hits = store.search("X", k=10, agent_id="a")
        assert {h.turn_id for h in hits} == {"t1"}
