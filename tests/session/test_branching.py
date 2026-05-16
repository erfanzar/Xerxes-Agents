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
"""Tests for xerxes.session.branching."""

from __future__ import annotations

import pytest
from xerxes.session.branching import branch_session, lineage
from xerxes.session.models import SessionRecord, TurnRecord
from xerxes.session.store import FileSessionStore


def _populate(store):
    rec = SessionRecord(
        session_id="root",
        agent_id="coder",
        turns=[TurnRecord(turn_id="t1", prompt="hello")],
    )
    store.save_session(rec)
    return rec


def test_branch_creates_child(tmp_path):
    store = FileSessionStore(tmp_path)
    _populate(store)
    child = branch_session(store, source_session_id="root", title="experiment")
    assert child.parent_session_id == "root"
    assert child.metadata["forked_from"] == "root"
    assert child.metadata["title"] == "experiment"
    # History was deep-copied.
    assert [t.turn_id for t in child.turns] == ["t1"]


def test_branch_does_not_mutate_source(tmp_path):
    store = FileSessionStore(tmp_path)
    _populate(store)
    child = branch_session(store, source_session_id="root")
    # Edit the child's turns; source must be unaffected.
    child.turns[0].prompt = "mutated"
    store.save_session(child)
    reread = store.load_session("root")
    assert reread.turns[0].prompt == "hello"


def test_branch_unknown_source_raises(tmp_path):
    store = FileSessionStore(tmp_path)
    with pytest.raises(KeyError):
        branch_session(store, source_session_id="ghost")


def test_lineage(tmp_path):
    store = FileSessionStore(tmp_path)
    _populate(store)
    branch_session(store, source_session_id="root", new_session_id="c1")
    branch_session(store, source_session_id="c1", new_session_id="g1")
    chain = lineage(store, "g1")
    assert chain == ["g1", "c1", "root"]
