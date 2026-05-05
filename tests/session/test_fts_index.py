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
"""Tests for xerxes.session.fts_index."""

from pathlib import Path

import pytest
from xerxes.session.fts_index import SessionFTSIndex
from xerxes.session.models import SessionRecord, TurnRecord


class TestSessionFTSIndex:
    def test_index_and_search(self, tmp_path: Path):
        db = tmp_path / "fts.db"
        index = SessionFTSIndex(db)

        if not index._fts_available:
            pytest.skip("FTS5 not available in this SQLite build")

        session = SessionRecord(
            session_id="s1",
            turns=[
                TurnRecord(
                    turn_id="t1",
                    prompt="How do I deploy to Kubernetes?",
                    response_content="Use kubectl apply -f deployment.yaml",
                ),
                TurnRecord(
                    turn_id="t2",
                    prompt="What about Docker?",
                    response_content="Use docker-compose up",
                ),
            ],
        )
        index.index_session(session)

        results = index.search("Kubernetes")
        assert len(results) == 1
        assert results[0]["session_id"] == "s1"
        assert "Kubernetes" in results[0]["content"]

    def test_search_multiple_sessions(self, tmp_path: Path):
        db = tmp_path / "fts.db"
        index = SessionFTSIndex(db)

        if not index._fts_available:
            pytest.skip("FTS5 not available in this SQLite build")

        index.index_session(
            SessionRecord(
                session_id="s1",
                turns=[TurnRecord(turn_id="t1", prompt="Python question", response_content="Answer")],
            )
        )
        index.index_session(
            SessionRecord(
                session_id="s2",
                turns=[TurnRecord(turn_id="t1", prompt="Python question again", response_content="Another answer")],
            )
        )

        results = index.search("Python")
        assert len(results) == 2

    def test_delete_session(self, tmp_path: Path):
        db = tmp_path / "fts.db"
        index = SessionFTSIndex(db)

        if not index._fts_available:
            pytest.skip("FTS5 not available in this SQLite build")

        index.index_session(
            SessionRecord(
                session_id="s1",
                turns=[TurnRecord(turn_id="t1", prompt="Secret", response_content="Data")],
            )
        )
        index.delete_session("s1")
        results = index.search("Secret")
        assert len(results) == 0

    def test_session_filter(self, tmp_path: Path):
        db = tmp_path / "fts.db"
        index = SessionFTSIndex(db)

        if not index._fts_available:
            pytest.skip("FTS5 not available in this SQLite build")

        index.index_session(
            SessionRecord(
                session_id="s1",
                turns=[TurnRecord(turn_id="t1", prompt="Hello world", response_content="Hi")],
            )
        )
        index.index_session(
            SessionRecord(
                session_id="s2",
                turns=[TurnRecord(turn_id="t1", prompt="Hello world", response_content="Hi")],
            )
        )

        results = index.search("world", session_id="s1")
        assert len(results) == 1
        assert results[0]["session_id"] == "s1"

    def test_empty_query_returns_empty(self, tmp_path: Path):
        db = tmp_path / "fts.db"
        index = SessionFTSIndex(db)
        assert index.search("") == []
        assert index.search("   ") == []
