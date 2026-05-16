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
"""Tests for xerxes.session.snapshots."""

from __future__ import annotations

import shutil

import pytest
from xerxes.session.snapshots import SnapshotManager


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "work"
    ws.mkdir()
    (ws / "a.txt").write_text("hello v1")
    return ws


@pytest.fixture
def shadow(tmp_path):
    s = tmp_path / "shadow"
    s.mkdir()
    return s


@pytest.fixture
def mgr(workspace, shadow):
    if shutil.which("git") is None:
        pytest.skip("git not on PATH")
    return SnapshotManager(workspace, shadow_root=shadow)


def test_snapshot_returns_record(mgr):
    rec = mgr.snapshot("first")
    assert rec.label == "first"
    assert rec.commit_sha
    assert rec.workspace_dir.endswith("work")


def test_list_includes_recent(mgr):
    a = mgr.snapshot("first")
    b = mgr.snapshot("second")
    listed = mgr.list()
    ids = [r.id for r in listed]
    assert a.id in ids
    assert b.id in ids


def test_get_by_id_label_sha(mgr):
    rec = mgr.snapshot("label-x")
    assert mgr.get(rec.id) is not None
    assert mgr.get("label-x") is not None
    assert mgr.get(rec.commit_sha[:7]) is not None


def test_rollback_restores_file(mgr, workspace):
    mgr.snapshot("v1")  # state: a.txt = "hello v1"
    (workspace / "a.txt").write_text("CHANGED")
    rec = mgr.list()[-1]
    mgr.rollback(rec.id)
    assert (workspace / "a.txt").read_text() == "hello v1"


def test_rollback_missing_raises(mgr):
    with pytest.raises(KeyError):
        mgr.rollback("does-not-exist")


def test_prune_keeps_recent(mgr):
    for _ in range(5):
        mgr.snapshot()
    removed = mgr.prune(keep=2)
    assert removed == 3
    assert len(mgr.list()) == 2


def test_reset_clears_shadow(mgr):
    mgr.snapshot("v1")
    mgr.reset()
    # The records file is gone; list() now returns [].
    assert mgr.list() == []
